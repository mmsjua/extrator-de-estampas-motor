from app import app, load_image, png_bytes
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import Response
from io import BytesIO
from PIL import Image
import numpy as np
import zipfile
import vtracer

@app.post("/extract-shirt-design")
async def extract_shirt_design(file: UploadFile = File(...)):
    """Experimental print-vs-fabric extraction inspired by tshirt-design-extractor.
    Uses dominant garment color; does not claim faithful recovery."""
    image = load_image(await file.read())
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w, _ = rgb.shape
    # Estimate fabric color from perimeter, where fabric usually dominates.
    border = np.concatenate([rgb[0,:,:],rgb[-1,:,:],rgb[:,0,:],rgb[:,-1,:]])
    q = (border//24).clip(0,10)
    keys, counts = np.unique(q,axis=0,return_counts=True)
    base = keys[counts.argmax()]*24+12
    distance = np.max(np.abs(rgb-base),axis=2)
    mask = distance > 48
    # Preserve existing alpha when present.
    mask &= np.asarray(image.getchannel("A"))>16
    out = np.asarray(image).copy()
    out[:,:,3] = np.where(mask,out[:,:,3],0).astype(np.uint8)
    result = Image.fromarray(out,"RGBA")
    png = png_bytes(result)
    archive = BytesIO()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("estampa_aproximada.png",png)
        try:
            svg=vtracer.convert_raw_image_to_svg(png,img_format="png",colormode="color",mode="spline")
            z.writestr("estampa_aproximada.svg",svg)
        except Exception:pass
        z.writestr("AVISO.txt","Metodo experimental por cor dominante. Falha em camisas estampadas, sombras e arte com cor semelhante ao tecido. Revise antes de imprimir.")
    return Response(archive.getvalue(),media_type="application/zip",headers={"Content-Disposition":'attachment; filename="estampa_teste.zip"'})
