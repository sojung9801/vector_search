export const VIDEO_FINDER_DEFAULT_BASE_URL = 'http://127.0.0.1:8787';

export const VIDEO_FINDER_DEFAULT_K = 8;

export const VIDEO_FINDER_MAX_K = 50;

export const VIDEO_FINDER_PATHS = {
  videos: '/api/videos',
  analyses: '/api/analyses',
} as const;

/** 분석 진행 상황 조회 간격 */
export const VIDEO_FINDER_ANALYSIS_POLL_MS = 2000;

export const VIDEO_FINDER_ANALYSIS_FINISHED = ['done', 'failed'] as const;
