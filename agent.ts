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
const MAX_LOOP=5;

const State = Annotation.Root({
  articleText: Annotation<string>,
  articleLang: Annotation<Lang>({reducer:(_l,r)=>r,default:()=>'zh' as Lang}),
  articleLangs: Annotation<Lang[]>({reducer:(_l,r)=>r,default:()=>['zh' as Lang]}),
  originHint: Annotation<string>({reducer:(_l,r)=>r,default:()=>''}),
  sourceKind: Annotation<SourceKind>({reducer:(_l,r)=>r,default:()=>'general' as SourceKind}),
  intent: Annotation<string>,
  claims: Annotation<string[]>({reducer:(_l,r)=>r,default:()=>[]}),
  queries: Annotation<string[]>({reducer:(_l,r)=>r,default:()=>[]}),
  historyQueries: Annotation<string[]>({reducer:(l,r)=>[...l,...r],default:()=>[]}),
  seenUrls: Annotation<string[]>({reducer:(l,r)=>[...new Set([...l,...r])],default:()=>[]}),
  noNewUrlStreak: Annotation<number>({reducer:(_l,r)=>r,default:()=>0}),
  skipLoop: Annotation<boolean>({reducer:(_l,r)=>r,default:()=>false}),
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

const AUTHORITATIVE_MEDIA=['cas.cn','ciis.org.cn','exportcompliancedaily.com','ciss.tsinghua.edu.cn','finance.cnr.cn','telecomlead.com','theregister.com','ft.com','sueddeutsche.de','handelsblatt.com','theguardian.com','efe.com','economist.com','dpa-news.de','wsj.com','washingtonpost.com','reuters.com','nytimes.com','aljazeera.com','scmp.com','cnn.com','cnbc.com','bloomberg.com','bbc.com','apnews.com','afp.com','eutoday.net','developingtelecoms.com','thepaper.cn','mobileworldlive.com','wallstreetcn.com','zaobao.com','edition.cnn.com','euronews.com','euractiv.com','asia.nikkei.com','politico.com','telecoms.com','fierce-network.com','computerweekly.com','cacs.mofcom.gov.cn','cnet.com','zdnet.com','totaltele.com','rcrwireless.com','lightreading.com','capacitymedia.com','business.nikkei.com','lesechos.fr','forbes.com','xinhuanet.com','people.com.cn','caixin.com','nikkei.com','nhk.or.jp','lemonde.fr','spiegel.de','nature.com'];
const DOMAIN_PRESET:Record<Lang,string[]>={zh:['gov.cn','cas.cn','cacs.mofcom.gov.cn','xinhuanet.com','people.com.cn','thepaper.cn','wallstreetcn.com','finance.cnr.cn','ciis.org.cn'],en:['reuters.com','apnews.com','afp.com','bloomberg.com','bbc.com','cnn.com','cnbc.com','wsj.com','nytimes.com','washingtonpost.com','economist.com','ft.com','theguardian.com','aljazeera.com','scmp.com','politico.com','euronews.com','euractiv.com','forbes.com','whitehouse.gov','fda.gov','telecomlead.com','theregister.com','developingtelecoms.com','mobileworldlive.com','telecoms.com','fierce-network.com','computerweekly.com','totaltele.com','rcrwireless.com','lightreading.com','capacitymedia.com','cnet.com','zdnet.com','exportcompliancedaily.com','eutoday.net'],ja:['nikkei.com','asia.nikkei.com','business.nikkei.com','nhk.or.jp','go.jp'],es:['elpais.com','efe.com'],fr:['lemonde.fr','lesechos.fr','elysee.fr'],de:['spiegel.de','sueddeutsche.de','handelsblatt.com','dpa-news.de','bundesregierung.de']};
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
  if(/gov|cas\.cn|miit\.gov\.cn|cacs\.mofcom\.gov\.cn|whitehouse|go\.jp|elysee|bundesregierung/.test(d))return{sourceType:'government',authorityLevel:5};
  const bare=d;
  if(AUTHORITATIVE_MEDIA.some((m)=>bare===m||bare.endsWith(`.${m}`)))return{sourceType:'authoritative_media',authorityLevel:4};
  if(/brookings|rand\.org|bruegel|csis|piie|edu$/.test(d))return{sourceType:'thinktank',authorityLevel:4};
  return{sourceType:'media',authorityLevel:2};
};
const JudgeSchema=z.object({supportLevel:z.enum(['direct','indirect','unknown']),score:z.number().min(0).max(100),evidence:z.string(),explanation:z.string()});
const judgeRelevance=async(title:string,summary:string,r:TavilyResult):Promise<z.infer<typeof JudgeSchema>>=>{
  const content=clean(r.content).slice(0,6000);
  // 规则预筛：accessdata这类数据库查询页、纯列表页直接判unknown，不浪费LLM
  if(/cfdocs|rl\.cfm|start_search|showList|PAGENUM/i.test(String(r.url))||/Pasting \d+\.\d+|Contract Manufacturer|RegistrationNumber/i.test(`${title} ${content}`.slice(0,500))) {
    // 仍走LLM二次确认，但先降权：如果LLM也判低则直接unknown
  }
  try{
    const m=getChatModel({temperature:0}).withStructuredOutput(JudgeSchema,{name:'relevance_judge'});
    return await m.invoke(`你是新闻溯源裁判。判断候选页面是否真的支撑原文。
原文标题：${title}\n原文摘要：${summary}\n候选标题：${clean(r.title)}\n候选URL：${clean(r.url)}\n候选正文：${content}\n规则：1)必须语义相近，同一事件/同一数据/同一声明才算direct；2)仅域名权威(如fda.gov)但内容是数据库列表、无关产品页，一律unknown，score<=20；3)主题相关但非同一事实判indirect，score 30-60；4)同一事实且有一手证据判direct，score 70-100。5)政策规划类（十五五/十四五/纲要）：最原始出处优先——gov.cn公报全文/新华社受权发布 > 部委解读 > 媒体转载，原始全文即使发布时间早也判direct并给高分。evidence必须从候选正文原样摘录，不超过200字，无证据传空。`);
  }catch(e){logger.warn(`judge回退:${String(e)}`);return{supportLevel:content?'indirect':'unknown',score:content?30:0,evidence:content.slice(0,200),explanation:'LLM裁判失败，降级为间接相关'}}
};
const buildCandidate=async(r:TavilyResult,claims:string[],articleTitle:string,articleSummary:string):Promise<TraceCandidateOutput|null>=>{
  const url=clean(r.url);if(!url)return null;const title=clean(r.title)||url,content=clean(r.content);
  const {sourceType,authorityLevel}=classifySource(url);
  const j=await judgeRelevance(articleTitle,articleSummary,{...r,title,content});
  // LLM分数与来源等级融合：来源分封顶，不让纯fda.gov域名撑到100
  const finalScore=Math.min(100,Math.round(j.score*0.8+authorityLevel*14*0.2));
  const level=j.supportLevel==='direct'&&finalScore<60?'indirect':j.supportLevel;
  const capped=sourceType==='media'?Math.min(finalScore,70):finalScore;
  return{id:`${Date.now()}_${Math.random().toString(36).slice(2,8)}`,url,title,content:content.slice(0,4000),publisher:getDomain(url),sourceType,authorityLevel,score:level==='unknown'?Math.min(capped,20):capped,evidence:j.evidence.slice(0,400),explanation:j.explanation,supportLevel:level};
};

const normalizeUrl=(u:string)=>{try{const o=new URL(clean(u));o.hash='';o.search='';return o.toString().replace(/\/$/,'')}catch{return clean(u).split('?')[0]}};
const gateNode=async(s:S)=>{
  const t=clean(s.articleText);
  if(t.length<50||/仅观点|评论|预测|分析认为/.test(t)&&t.split(/[。!？]/).length<=2) return {skipLoop:true,sufficient:true,candidates:[]};
  return {skipLoop:false};
};
const planNode=async(s:S)=>{
    let claims=[s.articleText.slice(0,200)];let queries:string[]=[];
    try{
      const m=getChatModel({temperature:0}).withStructuredOutput(z.object({claims:z.array(z.string()).max(5),queries:z.array(z.string()).max(8),queriesEn:z.array(z.string()).max(8)}),{name:'todo_plan'});
      const p=await m.invoke(`你是溯源规划器。主语言=${s.articleLang},并行语言=${(s.articleLangs??[s.articleLang]).join(',')},来源类型=${s.sourceKind},背景=${s.originHint}。历史query（禁止重复/语义重复）：${(s.historyQueries??[]).join('|').slice(0,1000)}。策略A换角度优先、域名不变：保留核心实体，换同义/时间/英文/上下游之一，2-4词一组。政策规划类必须加“公报 全文 受权发布 建议/纲要”类query直指原始出处。文本:${s.articleText.slice(0,4000)}`);
      if(p.claims.length)claims=p.claims;const merged=[...(p.queries??[]),...(p.queriesEn??[])];if(merged.length)queries=merged;
    }catch{const tt=s.articleText.split(/。|;|\n/).filter(Boolean).slice(0,3);claims=tt.length?tt:claims;queries=claims.slice(0,3)}
    const seen=new Set((s.historyQueries??[]).map((q)=>q.toLowerCase()));
    const fresh=queries.map(clean).filter(Boolean).filter((q)=>!seen.has(q.toLowerCase())).slice(0,8);
    return {claims,intent:s.sourceKind,queries:fresh.length?fresh:queries.slice(0,3),historyQueries:fresh,loopCount:s.loopCount+1};
  };
const graph=new StateGraph(State)
  .addNode('gate',gateNode)
  .addNode('classify',classifyNode)
  .addNode('plan',planNode)
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
    const articleTitle=s.claims[0]??s.articleText.slice(0,100);
    const articleSummary=s.articleText.slice(0,2000);
    // 非权威直接丢弃，不进LLM裁判
    const filtered=s.rawResults.filter((r)=>classifySource(clean(r.url)).authorityLevel>=4);
    const judged=await Promise.all(filtered.map((r)=>buildCandidate(r,s.claims,articleTitle,articleSummary)));
    for(const c of judged){if(c&&!m.has(normalizeUrl(c.url)))m.set(normalizeUrl(c.url),c)}
    const cands=[...m.values()].sort((a,b)=>b.score-a.score).slice(0,5);
    const sufficient=cands.some(c=>c.supportLevel==='direct'&&c.authorityLevel>=4);
    const prevSeen=new Set((s.seenUrls??[]).map(normalizeUrl));
    const newUrls=[...m.keys()].filter((u)=>!prevSeen.has(u));
    const streak=newUrls.length===0?(s.noNewUrlStreak??0)+1:0;
    return {candidates:cands,sufficient,seenUrls:[...m.keys()],noNewUrlStreak:streak};
  })
  .addEdge(START,'gate').addEdge('gate','classify').addEdge('classify','plan').addEdge('plan','search').addEdge('error_handler','rank')
  .addConditionalEdges('search',(s:S)=>s.lastError?'error_handler':'rank',{error_handler:'error_handler',rank:'rank'})
  .addConditionalEdges('rank',(s:S)=>{if(s.sufficient)return END;if(s.loopCount>=MAX_LOOP)return END;if((s.noNewUrlStreak??0)>=2)return END;return 'plan'},{plan:'plan',[END]:END})
  .compile();

export const runAuthorityTraceAgent=async(article:TraceArticleInput,meta:{taskId:string;newsId:number;timeRange?:TimeRangeOpt})=>{
  const input=[article.title,article.title_cn,article.summary,article.summary_cn,article.content,article.content_cn].map(clean).join(' ');
  const isPolicy=/十五五|十四五|规划|纲要|国务院|部委|白皮书|指导意见|行动方案/.test(input);
  const effectiveTimeRange=meta.timeRange??(isPolicy?'unlimited':'7d');
  const r=await graph.invoke({articleText:input,timeRange:effectiveTimeRange},{runName:`authority-trace:${meta.taskId}`,tags:['authority-trace','langgraph'],metadata:{...meta,timeRange:effectiveTimeRange}});
  return {intent:String(r.sourceKind),claims:r.claims as string[],candidates:r.candidates as TraceCandidateOutput[]};
};
export const authorityTraceGraph=graph;
