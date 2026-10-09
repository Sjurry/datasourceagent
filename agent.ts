import { tool } from '@langchain/core/tools';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { z } from 'zod';
import { getChatModel } from './aiChat/model.js';
import { logger } from './logger.js';

export interface TraceArticleInput { title: string|null; title_cn: string|null; content: string|null; content_cn: string|null; summary: string|null; summary_cn: string|null; }
export interface TraceCandidateOutput { id:string; url:string; title:string; content:string; publisher:string; sourceType:string; authorityLevel:number; score:number; evidence:string; explanation:string; supportLevel:'direct'|'indirect'|'unknown'; }
interface TavilyResult { title?:string; url?:string; content?:string; published_date?:string; score?:number; }
type Lang = 'zh'|'en'|'ja'|'es'|'fr'|'de';
type SourceKind = 'government'|'authoritative_media'|'thinktank'|'general';
const MAX_LOOP=4;

const State = Annotation.Root({
  articleText: Annotation<string>,
  articleLang: Annotation<Lang>({reducer:(_l,r)=>r,default:()=>'zh' as Lang}),
  articleLangs: Annotation<Lang[]>({reducer:(_l,r)=>r,default:()=>['zh' as Lang]}),
  originHint: Annotation<string>({reducer:(_l,r)=>r,default:()=>''}),
  sourceKind: Annotation<SourceKind>({reducer:(_l,r)=>r,default:()=>'general' as SourceKind}),
  intent: Annotation<string>,
  claims: Annotation<string[]>({reducer:(_l,r)=>r,default:()=>[]}),
  queries: Annotation<string[]>({reducer:(_l,r)=>r,default:()=>[]}),
  rawResults: Annotation<TavilyResult[]>({reducer:(l,r)=>[...l,...r],default:()=>[]}),
  candidates: Annotation<TraceCandidateOutput[]>({reducer:(_l,r)=>r,default:()=>[]}),
  loopCount: Annotation<number>({reducer:(_l,r)=>r,default:()=>0}),
  sufficient: Annotation<boolean>({reducer:(_l,r)=>r,default:()=>false}),
  lastError: Annotation<string|null>({reducer:(_l,r)=>r,default:()=>null}),
  retryCount: Annotation<number>({reducer:(_l,r)=>r,default:()=>0}),
  timeRange: Annotation<TimeRangeOpt>({reducer:(_l,r)=>r,default:()=>'7d' as TimeRangeOpt}),
});
type S=typeof State.State;
export type TimeRangeOpt='1d'|'3d'|'7d'|'30d'|'month'|'unlimited';
const timeRangeToTavily=(t:TimeRangeOpt):'day'|'week'|'month'|'year'|undefined=>t==='1d'?'day':t==='3d'||t==='7d'?'week':t==='30d'||t==='month'?'month':undefined;
const timeRangeCutoff=(t:TimeRangeOpt)=>t==='1d'?1:t==='3d'?3:t==='7d'?7:t==='30d'?30:Number.POSITIVE_INFINITY;

// —— 四类错误分级处理 ——
// 1. Transient(网络/限流/5xx/timeout):系统自动Retry,指数退避,最多3次
const isTransient=(e:unknown)=>{const m=String(e instanceof Error?e.message:String(e));return /429|5\d\d|timeout|ECONN|ETIMEDOUT|fetch failed|network/i.test(m)};
const withRetry=<T,>(fn:()=>Promise<T>,max=3)=>async():Promise<T>=>{let last:unknown;for(let i=0;i<max;i++){try{return await fn()}catch(e){last=e;if(!isTransient(e)||i===max-1)throw e;await new Promise(r=>setTimeout(r,500*2**i))}}throw last};
// 4. Unexpected:直接bubble up,不吞错(各节点不catch未知错误,仅处理已知两类)
const clean=(v:unknown)=>String(v??'').replace(/\s+/g,' ').trim();
const getDomain=(u:string)=>{try{return new URL(u).hostname.toLowerCase().replace(/^www\./,'')}catch{return''}};

// 规则+LLM二级classify:语言zh/en/ja/es/fr/de,来源government/authoritative_media/thinktank
const ruleLang=(t:string):Lang=>{
  if(/[\u3040-\u30ff]/.test(t))return 'ja';
  if(/[\u4e00-\u9fff]/.test(t))return 'zh';
  if(/[áéíóúñ¿¡]/.test(t))return 'es';
  if(/[àâçèêëîïôû]/.test(t))return 'fr';
  if(/[äöüß]/.test(t))return 'de';
  return 'en';
};
const ClassifySchema=z.object({lang:z.enum(['zh','en','ja','es','fr','de']),langs:z.array(z.enum(['zh','en','ja','es','fr','de'])).min(1).max(3),sourceKind:z.enum(['government','authoritative_media','thinktank','general']),originHint:z.string(),reason:z.string()});
const classifyNode=async(s:S)=>{
  let lang:Lang=ruleLang(s.articleText),langs:Lang[]=[lang],kind:SourceKind='general',originHint='';
  // 内容背景优先:英伟达/NVIDIA/OpenAI/Anthropic/SpaceX等→英文源;政策/规划/国务院→中文政府源
  if(/英伟达|NVIDIA|OpenShell|Sentry|BlueField|Anthropic|SpaceX|Scale AI|OpenAI|Google|Microsoft/i.test(s.articleText)){langs=['en','zh'];originHint='US-tech-company'}
  else if(/\.gov|政策|规划|国务院|部委|公告|gov\.cn|cas\.cn/.test(s.articleText)){langs=['zh'];originHint='CN-government';kind='government'}
  else if(/智库|研究所|brookings|rand|bruegel/i.test(s.articleText)){langs=['en','zh'];kind='thinktank'}
  try{
    const m=getChatModel({temperature:0}).withStructuredOutput(ClassifySchema,{name:'classify'});
    const r=await m.invoke(`你是溯源分类器。不要只看文字是中文就判中文站,要从内容背景判断原始出处。
规则:英伟达/NVIDIA/Anthropic/SpaceX/Scale AI/OpenShell/Sentry/BlueField等美国科技公司新闻→langs含en(主)+zh(辅),originHint=US-tech-company;中国政策/规划/部委→langs=[zh];日本企业/日元/日经→含ja。
返回langs按优先级排序(第一个为主语言)。文本:${s.articleText.slice(0,3000)}`);
    lang=r.lang;if(r.langs?.length)langs=r.langs as Lang[];kind=r.sourceKind;originHint=r.originHint;
  }catch(e){logger.warn(`classify回退:${String(e)}`)}
  return {articleLang:lang,articleLangs:langs,sourceKind:kind,originHint};
};

const DOMAIN_PRESET:Record<Lang,string[]>={zh:['gov.cn','cas.cn','xinhuanet.com','people.com.cn'],en:['reuters.com','bloomberg.com','whitehouse.gov','fda.gov'],ja:['nikkei.com','nhk.or.jp','go.jp'],es:['elpais.com','efe.com'],fr:['lemonde.fr','elysee.fr'],de:['spiegel.de','bundesregierung.de']};
const THINKTANK_DOMAINS=['brookings.edu','rand.org','bruegel.org','csis.org','piie.com'];

const tavilySearch=async(query:string,o:{topic:'news'|'general';maxResults?:number;includeDomains?:string[];timeRange?:TimeRangeOpt}):Promise<TavilyResult[]>=>{
  const tr=timeRangeToTavily(o.timeRange??'7d');
  const r=await fetch('https://api.tavily.com/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key:process.env.TAVILY_API_KEY,query,topic:o.topic,search_depth:'advanced',max_results:o.maxResults??8,...(o.includeDomains?.length?{include_domains:o.includeDomains}:{}),...(tr?{time_range:tr}:{}),include_answer:false}),signal:AbortSignal.timeout(30000)});
  if(!r.ok)throw new Error(`Tavily HTTP ${r.status}`);
  const d=await r.json() as {results?:TavilyResult[]};
  const cutoff=timeRangeCutoff(o.timeRange??'7d');if(!Number.isFinite(cutoff))return d.results??[];
  const now=Date.now();return (d.results??[]).filter(x=>{if(!x.published_date)return true;const t=Date.parse(x.published_date);return Number.isNaN(t)||(now-t)<=cutoff*864e5});
};
// 指定域名+全网双路配合,按语言组装查询
const searchBoth=tool(async({query,lang,sourceKind,timeRange}:{query:string;lang:Lang;sourceKind:SourceKind;timeRange:TimeRangeOpt})=>{
  const domains=sourceKind==='thinktank'?THINKTANK_DOMAINS:DOMAIN_PRESET[lang]??[];
  const [a,b]=await Promise.all([
    tavilySearch(query,{topic:'general',maxResults:5,includeDomains:domains,timeRange}).catch(()=>[] as TavilyResult[]),
    tavilySearch(query,{topic:query.length<30?'news':'general',maxResults:8,timeRange}).catch(()=>[] as TavilyResult[]),
  ]);
  const seen=new Map<string,TavilyResult>();
  for(const it of [...a,...b]){const u=clean(it.url);if(u&&!seen.has(u))seen.set(u,it)}
  return JSON.stringify([...seen.values()]);
},{name:'search_authoritative_sources',description:'指定域名搜索+全网搜索双路召回',schema:z.object({query:z.string().min(1),lang:z.string(),sourceKind:z.string(),timeRange:z.string().optional()})});

const classifySource=(url:string)=>{
  const d=getDomain(url);if(!d)return{sourceType:'unknown',authorityLevel:1};
  if(/gov|cas\.cn|miit\.gov\.cn|whitehouse|go\.jp|elysee|bundesregierung/.test(d))return{sourceType:'government',authorityLevel:5};
  if(/reuters|bloomberg|xinhua|people\.com|nikkei|nhk|lemonde|spiegel|caixin|nature/.test(d))return{sourceType:'authoritative_media',authorityLevel:4};
  if(/brookings|rand\.org|bruegel|csis|piie|edu$/.test(d))return{sourceType:'thinktank',authorityLevel:4};
  return{sourceType:'media',authorityLevel:2};
};
const buildCandidate=(r:TavilyResult,claims:string[]):TraceCandidateOutput|null=>{
  const url=clean(r.url);if(!url)return null;const title=clean(r.title)||url,content=clean(r.content);
  const {sourceType,authorityLevel}=classifySource(url);
  const matched=claims.some(c=>{const w=c.split(/[\s，。、“”‘’；：:,.!?]+/).filter(x=>x.length>=2);return w.length&&w.filter(x=>`${title} ${content}`.includes(x)).length>=Math.min(2,w.length)});
  const direct=matched&&content.length>80;
  return{id:`${Date.now()}_${Math.random().toString(36).slice(2,8)}`,url,title,content:content.slice(0,4000),publisher:getDomain(url),sourceType,authorityLevel,score:Math.min(100,authorityLevel*14+(direct?35:12)),evidence:content.slice(0,300),explanation:direct?'直接匹配':'间接相关',supportLevel:direct?'direct':content?'indirect':'unknown'};
};

const graph=new StateGraph(State)
  .addNode('classify',classifyNode)
  .addNode('plan',async(s:S)=>{
    // todo规划:按语言生成中英双语query
    let claims=[s.articleText.slice(0,200)];let queries:string[]=[];
    try{
      const m=getChatModel({temperature:0}).withStructuredOutput(z.object({claims:z.array(z.string()).max(5),queries:z.array(z.string()).max(8),queriesEn:z.array(z.string()).max(8)}),{name:'todo_plan'});
      const p=await m.invoke(`你是溯源规划器(TODO)。主语言=${s.articleLang},并行语言=${(s.articleLangs??[s.articleLang]).join(',')},来源类型=${s.sourceKind},背景=${s.originHint}。拆原子事实,queries给原文语言query,queriesEn给英文翻译query(英伟达/NVIDIA/OpenShell/Sentry/BlueField等专有名词必须保留英文原文)。文本:${s.articleText.slice(0,4000)}`);
      if(p.claims.length)claims=p.claims;const merged=[...(p.queries??[]),...(p.queriesEn??[])];if(merged.length)queries=merged;
    }catch{const t=s.articleText.split(/。|;|\n/).filter(Boolean).slice(0,3);claims=t.length?t:claims;queries=claims.slice(0,3)}
    return {claims,intent:s.sourceKind,queries:queries.length?queries:claims,loopCount:s.loopCount+1};
  })
  .addNode('search',async(s:S)=>{
    const out:TavilyResult[]=[];
    try{
      for(const q of s.queries.slice(0,8)){
        const qLang=/[\u4e00-\u9fff]/.test(q)?s.articleLang:'en' as Lang;
        // 1. Transient由系统自动重试;2. LLM-recoverable(tool失败/解析失败)抛给上层由rank/plan感知
        const raw=await withRetry(()=>searchBoth.invoke({query:q,lang:qLang,sourceKind:s.sourceKind,timeRange:s.timeRange}))();
        try{out.push(...JSON.parse(String(raw)) as TavilyResult[])}
        catch(parseErr){throw new Error(`PARSE_ERROR: ${String(parseErr)} query=${q}`)}
      }
      return {rawResults:out,lastError:null,retryCount:0};
    }catch(e){
      if(isTransient(e)){throw e} // 重试已耗尽仍是transient -> 走3. error_handler补偿分支,不吞
      // 2. LLM-recoverable:错误存入state,回环给plan让LLM调整query策略
      return {rawResults:[],lastError:String(e instanceof Error?e.message:String(e))};
    }
  })
  .addNode('error_handler',async(s:S)=>{
    // 3. Developer声明式补偿:重试耗尽/双路全失败后的降级——放宽域名做一次全网兜底
    logger.warn(`[authority-trace] error_handler补偿 loop=${s.loopCount} err=${s.lastError}`);
    try{
      const fallback=await tavilySearch(s.claims[0]??s.articleText.slice(0,60),{topic:'general',maxResults:5,timeRange:s.timeRange});
      return {rawResults:fallback,lastError:null};
    }catch(e){throw e} // 补偿也失败 -> 4. bubble up给开发者排查
  })
  .addNode('rank',async(s:S)=>{
    const m=new Map<string,TraceCandidateOutput>();
    for(const r of s.rawResults){const c=buildCandidate(r,s.claims);if(c&&!m.has(c.url))m.set(c.url,c)}
    const cands=[...m.values()].sort((a,b)=>b.score-a.score).slice(0,5);
    const sufficient=cands.some(c=>c.supportLevel==='direct'&&c.authorityLevel>=4); // 溯源到权威媒体/政府即停
    return {candidates:cands,sufficient};
  })
  .addEdge(START,'classify').addEdge('classify','plan').addEdge('plan','search').addEdge('error_handler','rank')
  .addConditionalEdges('search',(s:S)=>s.lastError?'error_handler':'rank',{error_handler:'error_handler',rank:'rank'})
  .addConditionalEdges('rank',(s:S)=>{if(s.sufficient)return END;if(s.loopCount>=MAX_LOOP)return END;return 'plan'},{plan:'plan',[END]:END})
  .compile();

export const runAuthorityTraceAgent=async(article:TraceArticleInput,meta:{taskId:string;newsId:number;timeRange?:TimeRangeOpt})=>{
  const input=[article.title,article.title_cn,article.summary,article.summary_cn,article.content,article.content_cn].map(clean).join(' ');
  const r=await graph.invoke({articleText:input,timeRange:meta.timeRange??'7d'},{runName:`authority-trace:${meta.taskId}`,tags:['authority-trace','langgraph'],metadata:meta});
  return {intent:String(r.sourceKind),claims:r.claims as string[],candidates:r.candidates as TraceCandidateOutput[]};
};
export const authorityTraceGraph=graph;
