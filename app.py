from io import BytesIO
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, HTMLResponse
from PIL import Image
import numpy as np
import zipfile
import json
from pathlib import Path

app = FastAPI(title="Extrator de Estampas - Motor Experimental", version="0.2.0")

@app.get('/', response_class=HTMLResponse)
def home():
    return Path(__file__).with_name('index.html').read_text(encoding='utf-8')

MAX_BYTES = 15_000_000
MAX_PIXELS = 25_000_000

def load_image(data):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Arquivo maior que 15 MB")
    try:
        image = Image.open(BytesIO(data))
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("Formato não suportado")
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, "Imagem maior que 25 megapixels")
        image.load()
        return image.convert("RGBA")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Envie imagem PNG, JPEG ou WebP válida")

def png_bytes(image):
    out = BytesIO()
    image.save(out, format="PNG", dpi=(300, 300))
    return out.getvalue()

@app.get("/health")
def health():
    return {"status":"ok","version":"0.2.0","capabilities":["remove_white_background","extract_transparent_blocks","upscale_interpolation"],"reconstruction_ai":False,"mockup_extraction":False}

@app.post("/remove-white-background")
async def remove_white_background(file: UploadFile = File(...), threshold: int = Query(245,ge=0,le=255)):
    image = load_image(await file.read())
    rgba = np.asarray(image).copy()
    # Atenção: remove também detalhes brancos de logos.
    rgba[:,:,3][rgba[:,:,:3].min(axis=2) >= threshold] = 0
    return Response(png_bytes(Image.fromarray(rgba,"RGBA")), media_type="image/png",
                    headers={"Content-Disposition":'attachment; filename="arte_transparente.png"'})

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), factor: int = Query(2,ge=1,le=4)):
    image = load_image(await file.read())
    if image.width * image.height * factor * factor > MAX_PIXELS:
        raise HTTPException(413, "Imagem ampliada ultrapassa 25 megapixels")
    result = image.resize((image.width*factor,image.height*factor), Image.Resampling.LANCZOS)
    return Response(png_bytes(result),media_type="image/png",
                    headers={"Content-Disposition":'attachment; filename="arte_ampliada.png"',"X-Processing-Note":"Interpolacao; nao e reconstrucao vetorial ou IA"})

@app.post("/extract-blocks")
async def extract_blocks(file: UploadFile = File(...), alpha_threshold: int = Query(16,ge=1,le=255),
                         min_area: int = Query(30,ge=1,le=100000),
                         merge_gap: int = Query(12,ge=0,le=80)):
    image = load_image(await file.read())
    alpha = np.asarray(image.getchannel("A"))
    if np.all(alpha==255):
        raise HTTPException(422, "Imagem sem transparencia. Envie PNG com fundo transparente; nao removemos automaticamente fundo de mockup.")
    # Componentes conexos por linhas: evita bibliotecas pesadas e mantém detalhes de cada bloco.
    # Dilatação aproximada em grade reduzida une letras/partes próximas sem separar pétalas.
    h,w = alpha.shape
    step = max(1, (max(h,w)+1799)//1800)
    small = alpha[::step,::step] >= alpha_threshold
    gap = max(0,merge_gap//step)
    if gap:
        from PIL import ImageFilter
        mask_im = Image.fromarray((small*255).astype("uint8"),"L")
        radius = 2*(gap//2)+1
        if radius > 1:
            small = np.asarray(mask_im.filter(ImageFilter.MaxFilter(min(radius, 161)))) > 0
    H,W=small.shape
    visited=np.zeros((H,W),dtype=np.uint8)
    blocks=[]
    for y in range(H):
        for x in range(W):
            if not small[y,x] or visited[y,x]:
                continue
            q=deque([(y,x)]);visited[y,x]=1
            minx=maxx=x;miny=maxy=y;count=0
            while q:
                cy,cx=q.popleft();count+=1
                minx=min(minx,cx);maxx=max(maxx,cx)
                miny=min(miny,cy);maxy=max(maxy,cy)
                for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0<=ny<H and 0<=nx<W and small[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx]=1;q.append((ny,nx))
            if count*step*step < min_area:continue
            pad=merge_gap
            box=(max(0,minx*step-pad),max(0,miny*step-pad),
                 min(w,(maxx+1)*step+pad),min(h,(maxy+1)*step+pad))
            if np.any(alpha[box[1]:box[3],box[0]:box[2]] >= alpha_threshold):
                blocks.append(box)
    blocks.sort(key=lambda b:(b[1],b[0]))
    if len(blocks)>100:
        raise HTTPException(422, "Mais de 100 blocos. Aumente min_area ou merge_gap")
    output=BytesIO()
    manifest={"version":"0.2.0","block_count":len(blocks),"warning":"Segmentacao geometrica experimental; nao identifica semanticamente logos, patrocinadores ou estampas.","blocks":[]}
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for i,box in enumerate(blocks,1):
            crop=image.crop(box)
            name=f"bloco_{i:02d}.png"
            z.writestr(name,png_bytes(crop))
            manifest["blocks"].append({"file":name,"bbox":box})
        z.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    return Response(output.getvalue(),media_type="application/zip",
                    headers={"Content-Disposition":'attachment; filename="blocos_estampa.zip"'})


@app.post("/extract-color-art")
async def extract_color_art(file: UploadFile = File(...),
                            saturation: int = Query(75,ge=20,le=220),
                            min_area: int = Query(20,ge=1,le=10000)):
    """Heuristic for colored motifs on pale shirt mockups. NOT AI reconstruction."""
    image = load_image(await file.read())
    rgb = np.asarray(image.convert("RGB"),dtype=np.uint8)
    mx=rgb.max(axis=2).astype(np.int16)
    mn=rgb.min(axis=2).astype(np.int16)
    chroma=mx-mn
    # Saturated pixels only; dark/white artwork may be lost.
    mask=(chroma>=saturation)&(mx>=75)
    # Preserve original pixel colors and make non-selected pixels transparent.
    rgba=np.asarray(image).copy()
    rgba[:,:,3]=np.where(mask,rgba[:,:,3],0).astype(np.uint8)
    colored=Image.fromarray(rgba,"RGBA")
    # Separate nearby colored motifs into groups; reused transparent-block endpoint logic.
    alpha=np.asarray(colored.getchannel("A"))
    h,w=alpha.shape
    step=max(1,(max(h,w)+1799)//1800)
    small=(alpha[::step,::step]>=16)
    from PIL import ImageFilter
    small=np.asarray(Image.fromarray((small*255).astype("uint8"),"L").filter(ImageFilter.MaxFilter(13)))>0
    H,W=small.shape
    seen=np.zeros((H,W),dtype=np.uint8)
    boxes=[]
    for y in range(H):
        for x in range(W):
            if seen[y,x] or not small[y,x]:continue
            queue=deque([(y,x)]);seen[y,x]=1
            x0=x1=x;y0=y1=y;area=0
            while queue:
                yy,xx=queue.popleft();area+=1
                x0=min(x0,xx);x1=max(x1,xx);y0=min(y0,yy);y1=max(y1,yy)
                for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                    if 0<=ny<H and 0<=nx<W and small[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=1;queue.append((ny,nx))
            if area*step*step<min_area:continue
            box=(max(0,x0*step-10),max(0,y0*step-10),
                 min(w,(x1+1)*step+10),min(h,(y1+1)*step+10))
            boxes.append(box)
    boxes.sort(key=lambda b:(b[1],b[0]))
    if len(boxes)>150:raise HTTPException(422,"Muitos blocos; aumente saturacao ou min_area")
    out=BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("arte_colorida_total.png",png_bytes(colored))
        unique=[]
        rejected=[]
        for box in boxes:
            crop=colored.crop(box)
            alpha=np.asarray(crop.getchannel("A"))
            ys,xs=np.where(alpha>0)
            if not len(xs):
                continue
            crop=crop.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
            cw,ch=crop.size
            # Descarta fragmentos muito pequenos e tiras compridas; heuristica revisavel.
            if cw*ch<180 or min(cw,ch)<9:
                rejected.append({"bbox":box,"reason":"fragmento pequeno"})
                continue
            if (cw>ch*3.0 and cw>65) or (ch>cw*5 and ch>90):
                rejected.append({"bbox":box,"reason":"possivel faixa da roupa"})
                continue
            # Comparar a silhueta e as cores com tolerancia a tamanho e rotacao.
            # Nao garante reconhecimento semantico; o arquivo completo permanece no ZIP.
            from PIL import ImageOps
            tile=ImageOps.contain(crop,(48,48),Image.Resampling.LANCZOS)
            normalized=Image.new("RGBA",(48,48),(0,0,0,0))
            normalized.alpha_composite(tile,((48-tile.width)//2,(48-tile.height)//2))
            sample=np.asarray(normalized).astype(np.int16)
            sample[:,:,:3]*=(sample[:,:,3:4]>30)
            ratio=cw/max(1,ch)
            duplicate=False
            for r,prev in unique:
                if abs(np.log(max(r,0.01)/max(ratio,0.01)))>0.3:
                    continue
                alpha_diff=np.mean(np.abs(sample[:,:,3]-prev[:,:,3]))
                rgb_diff=np.mean(np.abs(sample[:,:,:3]-prev[:,:,:3]))
                if alpha_diff<24 and rgb_diff<30:
                    duplicate=True
                    break
            if duplicate:
                rejected.append({"bbox":box,"reason":"possivel duplicata"})
                continue
            unique.append((ratio,sample))
            z.writestr(f"arte_unica_{len(unique):02d}.png",png_bytes(crop))
        z.writestr("relatorio.json",json.dumps({"detectados":len(boxes),"exportados":len(unique),"descartados":rejected,"aviso":"Filtro aproximado; pode remover desenhos validos. Confira arte_colorida_total.png."},ensure_ascii=False,indent=2))
        z.writestr("LEIA-ME.txt","EXTRACAO EXPERIMENTAL POR COR. Pode perder textos pretos, brancos e detalhes pouco saturados, incluir detalhes da roupa e dividir ou juntar logos incorretamente. Nao remove perspectiva nem reconstrói arte. Confira antes de imprimir.")
    return Response(out.getvalue(),media_type="application/zip",
                    headers={"Content-Disposition":'attachment; filename="artes_coloridas_experimentais.zip"'})
