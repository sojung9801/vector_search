import {
  VIDEO_FINDER_DEFAULT_BASE_URL,
  VIDEO_FINDER_DEFAULT_K,
  VIDEO_FINDER_MAX_K,
  VIDEO_FINDER_PATHS,
} from './videoFinderConstants';
import type {
  VideoFinderAnalysis,
  VideoFinderAnalysisRequest,
  VideoFinderSearchParams,
  VideoFinderSearchResult,
  VideoFinderVideo,
} from './videoFinderTypes';

export class VideoFinderRequestError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
  ) {
    super(`VideoFinder request failed: ${status} ${path}`);
    this.name = 'VideoFinderRequestError';
  }
}

/**
 * 영상찾기 로컬 서버(vf-server) 클라이언트.
 *
 * const client = createVideoFinderClient();
 * await client.startAnalysis({ contentId, url });           // 상세 화면 '영상 분석하기'
 * const analysis = await client.getAnalysis(contentId);    // status 가 done 이 될 때까지 조회
 *
 * const [video] = await client.listVideos();
 * const result = await client.search({ videoId: video.id, query: '두 손을 번쩍 든 장면' });
 * videoRef.current.src = client.buildStreamUrl(video.id);
 * videoRef.current.currentTime = result.shots[0].start_ms / 1000;
 */
export const createVideoFinderClient = (baseUrl: string = VIDEO_FINDER_DEFAULT_BASE_URL) => {
  const root = baseUrl.replace(/\/+$/, '');
  const videoPath = (videoId: string) => `${VIDEO_FINDER_PATHS.videos}/${encodeURIComponent(videoId)}`;

  const getJson = async <T>(path: string, signal?: AbortSignal): Promise<T> => {
    const res = await fetch(`${root}${path}`, { signal });
    if (!res.ok) {
      throw new VideoFinderRequestError(res.status, path);
    }
    return (await res.json()) as T;
  };

  const postJson = async <T>(path: string, body: object, signal?: AbortSignal): Promise<T> => {
    const res = await fetch(`${root}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      throw new VideoFinderRequestError(res.status, path);
    }
    return (await res.json()) as T;
  };

  return {
    /** 분석 시작. 이미 진행 중이면 그 작업 상태를 돌려준다. */
    startAnalysis: (req: VideoFinderAnalysisRequest, signal?: AbortSignal) =>
      postJson<VideoFinderAnalysis>(
        VIDEO_FINDER_PATHS.analyses,
        { content_id: req.contentId, url: req.url ?? null, path: req.path ?? null },
        signal,
      ),

    /** 분석 상태. 요청한 적 없으면 404 → VideoFinderRequestError */
    getAnalysis: (contentId: string, signal?: AbortSignal) =>
      getJson<VideoFinderAnalysis>(`${VIDEO_FINDER_PATHS.analyses}/${encodeURIComponent(contentId)}`, signal),

    listVideos: (signal?: AbortSignal) => getJson<VideoFinderVideo[]>(VIDEO_FINDER_PATHS.videos, signal),

    getVideo: (videoId: string, signal?: AbortSignal) => getJson<VideoFinderVideo>(videoPath(videoId), signal),

    search: ({ videoId, query, k = VIDEO_FINDER_DEFAULT_K, signal }: VideoFinderSearchParams) => {
      const params = new URLSearchParams({ q: query, k: String(Math.min(Math.max(k, 1), VIDEO_FINDER_MAX_K)) });
      return getJson<VideoFinderSearchResult>(`${videoPath(videoId)}/search?${params}`, signal);
    },

    buildStreamUrl: (videoId: string) => `${root}${videoPath(videoId)}/stream`,

    buildThumbUrl: (videoId: string, thumb: string) => `${root}${videoPath(videoId)}/thumbs/${encodeURIComponent(thumb)}`,
  };
};

export type VideoFinderClient = ReturnType<typeof createVideoFinderClient>;
