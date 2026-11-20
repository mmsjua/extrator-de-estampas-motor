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
    return {"status":"ok","version":"0.3.0","capabilities":["crop_region","extract_transparent_blocks","vectorize_svg_vtracer"],"reconstruction_ai":False,"mockup_extraction":False}

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


@app.post("/crop-region")
async def crop_region(file: UploadFile = File(...),
                      x: int = Query(...,ge=0), y: int = Query(...,ge=0),
                      width: int = Query(...,ge=1), height: int = Query(...,ge=1),
                      mode: str = Query("original")):
    """Manual precise crop. Transparent color filtering is optional and imperfect."""
    image=load_image(await file.read())
    if x>=image.width or y>=image.height or x+width>image.width or y+height>image.height:
        raise HTTPException(422,"Area selecionada fora da imagem")
    crop=image.crop((x,y,x+width,y+height))
    if mode=="color":
        a=np.asarray(crop).copy()
        rgb=a[:,:,:3].astype(np.int16)
        chroma=rgb.max(axis=2)-rgb.min(axis=2)
        a[:,:,3]=np.where(chroma>=55,a[:,:,3],0).astype(np.uint8)
        crop=Image.fromarray(a,"RGBA")
    elif mode!="original":
        raise HTTPException(422,"Modo invalido")
    return Response(png_bytes(crop),media_type="image/png",
                    headers={"Content-Disposition":'attachment; filename="recorte_manual.png"'})

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    """VTracer SVG conversion of selected logo artwork. Not semantic reconstruction."""
    import vtracer
    image=load_image(await file.read())
    if image.width*image.height>4_000_000:
        raise HTTPException(413,"Para vetorizar, use recorte de ate 4 megapixels")
    data=png_bytes(image)
    try:
        svg=vtracer.convert_raw_image_to_svg(data,img_format="png",
                                            colormode="color",hierarchical="stacked",
                                            mode="spline",filter_speckle=4,
                                            color_precision=6,layer_difference=16)
    except Exception as exc:
        raise HTTPException(422,"Nao foi possivel vetorizar esta imagem") from exc
    return Response(svg,media_type="image/svg+xml",
                    headers={"Content-Disposition":'attachment; filename="arte_vetorizada.svg"'})


@app.post("/vectorize-package")
async def vectorize_package(file: UploadFile = File(...), scale: int = Query(4,ge=1,le=6)):
    """Vector paths + PNG rendered from SVG; similarity report, no invented AI reconstruction."""
    import vtracer
    import cairosvg
    from PIL import ImageChops,ImageStat
    image=load_image(await file.read())
    if image.width*image.height>2_000_000:
        raise HTTPException(413,"Recorte a logo para ate 2 megapixels antes de vetorizar")
    if image.width*image.height*scale*scale>25_000_000:
        raise HTTPException(413,"PNG ampliado excede limite de 25 megapixels")
    try:
        svg=vtracer.convert_raw_image_to_svg(png_bytes(image),img_format="png",
                                            colormode="color",hierarchical="stacked",
                                            mode="spline",filter_speckle=2,
                                            color_precision=8,layer_difference=8)
        preview=cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                                output_width=image.width,output_height=image.height)
        preview_img=Image.open(BytesIO(preview)).convert("RGBA")
        original_rgb=image.convert("RGB")
        traced_rgb=preview_img.convert("RGB")
        import numpy as np
        err=float(np.mean(np.abs(np.asarray(original_rgb,dtype=np.float32)-
                                 np.asarray(traced_rgb,dtype=np.float32))))
        png=cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                            output_width=image.width*scale,
                            output_height=image.height*scale)
        png_image=Image.open(BytesIO(png)).convert("RGBA")
        output=BytesIO()
        with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("arte_vetorizada.svg",svg)
            z.writestr("arte_vetorial_alta_resolucao.png",png_bytes(png_image))
            z.writestr("comparacao_original.png",png_bytes(image))
            z.writestr("relatorio_qualidade.json",json.dumps({
                "metodo":"VTracer + CairoSVG",
                "erro_medio_rgb_0_a_255":round(err,2),
                "resolucao_png":list(png_image.size),
                "escala":scale,
                "aviso":"A metrica compara pixels, nao garante fidelidade de letras nem reconstrucao de detalhes perdidos. O SVG pode conter partes de tecido se estiverem no recorte."
            },ensure_ascii=False,indent=2))
        return Response(output.getvalue(),media_type="application/zip",
                        headers={"Content-Disposition":'attachment; filename="vetor_e_png.zip"'})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422,"Falha na vetorizacao ou renderizacao SVG") from exc
