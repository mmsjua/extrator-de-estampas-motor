"""Smoke tests: transparent logo, vector output, zip contents and basic quality."""
from io import BytesIO
import json, zipfile
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from app import app

def test_vector_package_transparent_logo():
    image=Image.new("RGBA",(240,140),(0,0,0,0))
    draw=ImageDraw.Draw(image)
    draw.ellipse((20,20,120,120),fill=(235,40,20,255))
    draw.rectangle((135,35,220,105),fill=(30,80,220,255))
    raw=BytesIO();image.save(raw,format="PNG")
    response=TestClient(app).post("/vectorize-package?scale=4",
        files={"file":("logo.png",raw.getvalue(),"image/png")})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        assert {"arte_vetorizada.svg","arte_vetorial_alta_resolucao.png","relatorio_qualidade.json"}<=set(z.namelist())
        svg=z.read("arte_vetorizada.svg").decode()
        assert "<svg" in svg and "<path" in svg
        png=Image.open(BytesIO(z.read("arte_vetorial_alta_resolucao.png")))
        assert png.size==(960,560)
        report=json.loads(z.read("relatorio_qualidade.json"))
        assert report["erro_medio_rgb_0_a_255"]<80,report
