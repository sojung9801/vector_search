// 영상찾기 HTTP API 응답 타입 (app/api/server.py 와 맞춘다)

export type SearchMode = 'exact' | 'semantic';

/**
 * gesture: "왼손 드는"처럼 정해진 몸짓 질의 → 몸짓 결과가 주, 화면 결과는 접힘(좌우 구분 불가)
 * visual: "~장면"처럼 화면을 찾는 질의 → 화면 결과를 먼저 보여주고 발화는 접는다
 */
export type SearchIntent = 'gesture' | 'visual' | 'general';

export type HitKind = 'speech' | 'shot' | 'scene' | 'gesture';

export interface VideoFinderHit {
  kind: HitKind;
  start_ms: number;
  end_ms: number;
  /** 질의와 관련 있을 확률 추정치. 발화(재정렬 모델)와 화면(SigLIP2) 점수는 서로 비교할 수 없다. exact 모드에서는 1. */
  score: number;
  text: string | null;
  lang: string | null;
  /** 썸네일 파일명. buildThumbUrl 로 URL을 만든다. 발화 결과는 그 시점 직전 프레임. 프레임이 없는 색인이면 null */
  thumb: string | null;
  /** 검색어가 자막 원문에 그대로 포함됨 */
  exact: boolean;
  /** 접어서 보여줄 결과. 발화·장면 설명: 확률 0.05 미만, 화면: 0.02 미만이거나 말을 찾는 질의에 확실한 발화 결과가 있을 때 */
  low: boolean;
}

export interface VideoFinderSearchResult {
  video_id: string;
  query: string;
  mode: SearchMode;
  intent: SearchIntent;
  elapsed_ms: number;
  /** 관련 없는 결과는 서버에서 걸러지므로 빈 배열일 수 있다 */
  speech: VideoFinderHit[];
  shots: VideoFinderHit[];
  /** 구간 장면 설명(VLM) 검색 결과. text 는 "영어\n한국어". 장면 설명을 만든 영상에만 있다 */
  scenes: VideoFinderHit[];
  /** 사람 몸짓(관절 규칙, 실험) 구간. 검색어에 "왼손 드는" 같은 정해진 몸짓이 있을 때만. text 는 몸짓 이름 */
  gestures: VideoFinderHit[];
}

export interface VideoFinderVideo {
  /** 색인 id (원본 SHA-256 앞 16자) */
  id: string;
  name: string;
  duration_ms: number;
  speech_segments: number;
  shots: number;
  created_at: string | null;
}

export interface VideoFinderSearchParams {
  videoId: string;
  query: string;
  k?: number;
  signal?: AbortSignal;
}

export type AnalysisStatus = 'queued' | 'downloading' | 'indexing' | 'done' | 'failed';

export interface VideoFinderAnalysis {
  content_id: string;
  /** 분석에 쓴 url 또는 로컬 경로 */
  source: string;
  status: AnalysisStatus;
  /** 진행·실패 메시지 (서버 로그 문구, 표시용 아님) */
  message: string;
  /** status 가 done 이면 이 id로 search 한다 */
  video_id: string | null;
  updated_at: string;
}

/** url(비DRM mp4·webm·HLS) 또는 path(서버 Mac의 로컬 파일) 중 하나 */
export type VideoFinderAnalysisRequest =
  | { contentId: string; url: string; path?: never }
  | { contentId: string; path: string; url?: never };
