from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np

app = FastAPI(title="Extrator de Estampas - Motor Inicial", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "capabilities": ["remove_white_background"], "reconstruction_ai": False}

@app.post("/remove-white-background")
async def remove_white_background(file: UploadFile = File(...), threshold: int = 245):
    if not 0 <= threshold <= 255:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0 e 255")
    data = await file.read()
    if len(data) > 15_000_000:
        raise HTTPException(status_code=413, detail="Arquivo muito grande (max 15 MB)")
    try:
        image = Image.open(BytesIO(data))
        image.load()
        if image.width * image.height > 25_000_000:
            raise HTTPException(status_code=413, detail="Imagem muito grande")
        rgba = np.asarray(image.convert("RGBA")).copy()
        whiteness = rgba[:, :, :3].min(axis=2)
        rgba[:, :, 3][whiteness >= threshold] = 0
        output = Image.fromarray(rgba, "RGBA")
        buf = BytesIO()
        output.save(buf, format="PNG", dpi=(300, 300))
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Content-Disposition": 'attachment; filename="arte_transparente.png"'})
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Imagem inválida. Envie PNG ou JPEG.")
