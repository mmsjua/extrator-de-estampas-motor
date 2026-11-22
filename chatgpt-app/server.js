import express from "express";
import fs from "node:fs";
import sharp from "sharp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

const app = express();
app.use(express.json({limit:"16mb"}));
const html = fs.readFileSync(new URL("./widget.html",import.meta.url),"utf8");
const ops=["fundo","logo","frente","costas","fundo_logo","textos_artes","mangas","nova_imagem","selecionar_area"];
const limits={file:7_000_000,pixels:20_000_000};
const notice={
 fundo:"Recorte PNG gerado. Remoção inteligente de logos e textos ainda não implementada.",
 logo:"Recorte PNG fiel; a transparência original é preservada. Remoção inteligente do fundo ainda não implementada.",
 frente:"PNG recortado. Planificação de perspectiva ainda não implementada.",
 costas:"PNG recortado. Planificação de perspectiva ainda não implementada.",
 fundo_logo:"Recorte PNG gerado. Separação automática fundo/logo ainda não implementada.",
 textos_artes:"Recorte PNG fiel; a transparência original é preservada. Detecção de blocos ainda não implementada.",
 mangas:"PNG recortado. Selecione manualmente a região da manga.",
 selecionar_area:"Área selecionada exportada em PNG."
};
async function processImage({operacao,imagem_base64,area}){
  try{
   if(operacao==="nova_imagem")return {structuredContent:{ok:true,message:"Escolha outra imagem."},content:[{type:"text",text:"Escolha outra imagem."}]};
   const raw=imagem_base64.replace(/^data:image\/(png|jpeg|webp);base64,/,"");
   if(!/^[a-zA-Z0-9+/=]+$/.test(raw))throw Error("Imagem inválida.");
   const bytes=Buffer.from(raw,"base64");
   if(bytes.length>limits.file)throw Error("Imagem maior que 7 MB.");
   const info=await sharp(bytes).metadata();
   if(!info.width||!info.height||info.width*info.height>limits.pixels)throw Error("Dimensões excedem o limite.");
   let img=sharp(bytes).rotate();
   if(area&&area.w>0.005&&area.h>0.005){
    const left=Math.min(info.width-1,Math.floor(area.x*info.width));
    const top=Math.min(info.height-1,Math.floor(area.y*info.height));
    const width=Math.max(1,Math.min(info.width-left,Math.floor(area.w*info.width)));
    const height=Math.max(1,Math.min(info.height-top,Math.floor(area.h*info.height)));
    img=img.extract({left,top,width,height});
   }
   // Upscale determinístico ANTES da exportação final. Evita filtros criativos.
   // Limites protegem a memória do servidor gratuito e o tamanho da resposta MCP.
   const base=await img.ensureAlpha().png().toBuffer();
   const baseMeta=await sharp(base).metadata();
   const maxSide=Math.max(baseMeta.width,baseMeta.height);
   const scale=Math.min(4,3200/maxSide,Math.sqrt(12_000_000/(baseMeta.width*baseMeta.height)));
   const factor=Math.max(1,scale);
   const targetW=Math.max(1,Math.round(baseMeta.width*factor));
   const targetH=Math.max(1,Math.round(baseMeta.height*factor));
   let png=await sharp(base).resize(targetW,targetH,{kernel:"lanczos3",fit:"fill"})
    .withMetadata({density:300}).png({compressionLevel:9}).toBuffer();
   // Fidelidade: não apagar pixels brancos automaticamente; branco pode fazer parte da arte.
   // Transparência existente no original é preservada, mas fundo complexo não é segmentado.
   if(png.length>8_000_000)throw Error("PNG grande demais; selecione uma área menor.");
   return {structuredContent:{ok:true,filename:operacao+".png",png_base64:png.toString("base64"),notice:notice[operacao]+" Upscale "+factor.toFixed(2)+"x ("+targetW+"×"+targetH+" px), 300 DPI em metadados.",upscale:{factor,width:targetW,height:targetH,algorithm:"lanczos3_no_sharpen",ai:false}},content:[{type:"text",text:"PNG gerado. "+notice[operacao]}]};
  }catch(e){return {isError:true,structuredContent:{ok:false,error:e.message},content:[{type:"text",text:"Erro: "+e.message}]}};
 }
function makeServer(){
 const s=new McpServer({name:"extrator-de-estampas",version:"0.4.1"});
 s.resource("painel","ui://extrator/painel.html",async()=>({contents:[{uri:"ui://extrator/painel.html",mimeType:"text/html+skybridge",text:html}]}));
 s.tool("abrir_extrator","Abre painel interativo do Extrator de Estampas com 9 botões e upload.",{},async()=>({content:[{type:"text",text:"Painel de extração disponível."}],_meta:{"openai/outputTemplate":"ui://extrator/painel.html"}}));
 s.tool("processar_estampa","Exporta recorte PNG da imagem enviada; remove branco quase puro nas opções logo e textos. Não faz reconstrução por IA.",{
 operacao:z.enum(ops),imagem_base64:z.string().max(12_000_000),
 area:z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1),w:z.number().min(0).max(1),h:z.number().min(0).max(1)}).optional()
 },processImage);
 return s;
}
app.get("/health",(_,res)=>res.json({ok:true,version:"0.4.0",image_processing:"png_crop_upscale_lanczos3",ai_generation:false}));
app.get("/",(_,res)=>res.type("text").send("Extrator de Estampas MCP: conecte /mcp ao ChatGPT."));
app.use((req,res,next)=>{res.setHeader("Access-Control-Allow-Origin","https://mmsjua.github.io");res.setHeader("Access-Control-Allow-Methods","POST, OPTIONS");res.setHeader("Access-Control-Allow-Headers","Content-Type");if(req.method==="OPTIONS")return res.sendStatus(204);next()});
app.post("/api/processar",async(req,res)=>{try{const {operacao,imagem_base64,area}=req.body||{};if(!ops.includes(operacao)||typeof imagem_base64!=="string")return res.status(400).json({ok:false,error:"Operação ou imagem inválida"});const result=await processImage({operacao,imagem_base64,area});const data=result.structuredContent||{};res.status(data.ok?200:422).json(data)}catch(e){res.status(500).json({ok:false,error:String(e.message||e)})}});

const aiPrompts={
 fundo:"Extraia somente a padronagem de fundo da camisa da imagem de referência. Produza arte plana retangular para impressão, vista frontal, sem mockup, tecido, gola, costuras, dobras, sombras, logos, brasões, textos ou números. Preserve exatamente cores, motivos e disposição verificáveis. Não invente elementos nas partes ocultas. Imagem de alta nitidez.",
 logo:"Isole apenas a logomarca principal presente na camisa de referência. Preserve rigorosamente grafia, acentos, símbolos, cores, proporções e traços. Fundo transparente. Não redesenhe criativamente nem invente detalhes ilegíveis. Enquadramento justo, sem margem vazia.",
 frente:"Reproduza a arte plana da frente da camisa para impressão, sem corpo, gola, costuras, dobras, tecido ou perspectiva. Preserve fundo, logos e textos da referência nas posições verificáveis. Não invente letras ou detalhes ocultos.",
 costas:"Reproduza somente a arte plana das costas da camisa para impressão, sem mockup, tecido, costuras ou perspectiva. Preserve fielmente padronagem, letras, cores e logos visíveis; não invente elementos.",
 fundo_logo:"Extraia a padronagem plana de fundo da camisa, sem logomarcas nem textos. Preserve cores e motivos verificáveis. Não invente detalhes escondidos.",
 textos_artes:"Isole as letras e ilustrações da estampa da camisa, sem padronagem de fundo ou tecido. Fundo transparente. Preserve exatamente grafia, formas, proporções e cores visíveis, sem inventar.",
 mangas:"Reproduza a arte plana da manga visível na referência, sem tecido, costuras, perspectiva ou dobras. Preserve somente os elementos confirmados."
};
app.get("/api/ai-status",(_,res)=>res.json({configured:Boolean(process.env.OPENAI_API_KEY),model:process.env.OPENAI_IMAGE_MODEL||"gpt-image-1",requiresSeparateApiBilling:true}));
app.post("/api/ai-extract",async(req,res)=>{
 try{
  const {operacao,imagem_base64}=req.body||{};
  if(!Object.hasOwn(aiPrompts,operacao)||typeof imagem_base64!=="string")return res.status(400).json({ok:false,error:"Operação ou imagem inválida"});
  if(!process.env.OPENAI_API_KEY)return res.status(503).json({ok:false,error:"IA ainda não configurada no servidor. É necessária uma chave de API com faturamento próprio."});
  const raw=imagem_base64.replace(/^data:image\\/(png|jpeg|webp);base64,/,"");
  if(!/^[A-Za-z0-9+/=]+$/.test(raw))return res.status(400).json({ok:false,error:"Imagem inválida"});
  const bytes=Buffer.from(raw,"base64");
  if(bytes.length>7_000_000)return res.status(413).json({ok:false,error:"Imagem maior que 7 MB"});
  const meta=await sharp(bytes).metadata();
  if(!meta.width||!meta.height||meta.width*meta.height>20_000_000)return res.status(413).json({ok:false,error:"Dimensões excedem o limite"});
  const png=await sharp(bytes).rotate().png().toBuffer();
  const form=new FormData();
  form.append("model",process.env.OPENAI_IMAGE_MODEL||"gpt-image-1");
  form.append("image",new Blob([png],{type:"image/png"}),"referencia.png");
  form.append("prompt",aiPrompts[operacao]+" Não afirme fidelidade perfeita onde a referência é insuficiente.");
  form.append("size","1024x1536");
  form.append("quality","high");
  const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),180000);
  let response;try{response=await fetch("https://api.openai.com/v1/images/edits",{method:"POST",headers:{Authorization:"Bearer "+process.env.OPENAI_API_KEY},body:form,signal:controller.signal})}finally{clearTimeout(timer)}
  const data=await response.json();
  if(!response.ok)return res.status(502).json({ok:false,error:data.error?.message||"Falha no serviço de IA"});
  const b64=data.data?.[0]?.b64_json;
  if(!b64)return res.status(502).json({ok:false,error:"A API não retornou imagem"});
  return res.json({ok:true,filename:operacao+"_ia.png",png_base64:b64,notice:"Resultado gerado por IA. Compare grafia, símbolos e detalhes com a foto original antes de imprimir. Não é SVG vetorial."});
 }catch(e){return res.status(500).json({ok:false,error:e.name==="AbortError"?"Tempo limite de processamento de IA":String(e.message||e)})}
});

app.post("/mcp",async(req,res)=>{const s=makeServer();const t=new StreamableHTTPServerTransport({sessionIdGenerator:undefined});res.on("close",()=>{t.close();s.close()});try{await s.connect(t);await t.handleRequest(req,res,req.body)}catch(e){if(!res.headersSent)res.status(500).json({error:String(e)})}});
app.listen(process.env.PORT||3000,"0.0.0.0",()=>console.log("MCP pronto"));
