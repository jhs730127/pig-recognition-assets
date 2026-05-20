/**
 * PrerenderedTtsPlayer — Web Audio API 預生成 MP3 播放器。
 *
 * 給用 pig-recognition-assets 的預生成 TTS MP3（zh-TW, en-US 等）的新專案用。
 * 設計重點：
 *   1. **Web Audio API**（不是 HTMLAudioElement）— `decodeAudioData` + `start(when)`
 *      精準排程，組合句段與段間零 gap，比 `new Audio().play()` 體驗好很多。
 *   2. **Trim silence**：edge-tts MP3 每段前後有 200~500ms 靜音 padding，連續播會
 *      聽到「字字停頓」。decode 後找首尾低振幅 region 剪掉、留 25ms padding 保護
 *      中文入聲/輕聲尾音。
 *   3. **iOS / Chrome autoplay 解鎖**：setupAudioUnlock 註冊 user gesture listener，
 *      第一次 click/touchend/keydown 內 resume AudioContext 解鎖整個 session。
 *   4. **缺檔 fallback**：playPhrase / playSentence 回 false 時 caller 自己決定要不要
 *      走 Web Speech API fallback。
 *
 * 使用方式：
 *
 *   import { PrerenderedTtsPlayer } from './prerendered-tts-player';
 *
 *   const BASE = 'https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0';
 *   const player = new PrerenderedTtsPlayer({
 *     baseUrl: `${BASE}/audio/tts-zh-tw`,
 *     // manifestUrl 預設 = `${baseUrl}/manifest.json`
 *     enabled: () => mySettingsStore.speechEnabled,  // 可選；不傳一律啟用
 *   });
 *
 *   // App layout mount 時呼叫一次
 *   player.setupAudioUnlock();
 *   await player.loadManifest();
 *
 *   // 單句
 *   const ok = await player.playPhrase('feedback_correct_1');
 *   if (!ok) speechSynthesisFallback();
 *
 *   // 組合句（「5 減 1 等於多少」）
 *   await player.playSentence([
 *     PrerenderedTtsPlayer.numberKey(5, 'zh-TW')!,
 *     PrerenderedTtsPlayer.operationKey('-')!,
 *     PrerenderedTtsPlayer.numberKey(1, 'zh-TW')!,
 *     'q_equals_what',
 *   ]);
 *
 * Manifest 規格：見 audio/tts-zh-tw/manifest.json。
 */

export interface TtsManifest {
	voice: { zh: string; en: string };
	rate: string;
	phrases: Record<string, string>;
	count: number;
}

export interface PrerenderedTtsOptions {
	/** Audio 目錄（例：CDN URL 或 same-origin "/audio/tts"）。manifest 假設在 ${baseUrl}/manifest.json */
	baseUrl: string;
	/** 自訂 manifest URL（不傳時 = baseUrl + "/manifest.json"） */
	manifestUrl?: string;
	/** 啟用 getter；回傳 false 時 playPhrase/playSentence 一律回 false */
	enabled?: () => boolean;
	/**
	 * 播放速度（預設 1.0）。Web Audio API playbackRate 加速會自動 resample，
	 * 預設不保 pitch；想完全自然就用 1.0。
	 */
	playbackRate?: number;
	/** Trim silence 的 amplitude threshold（預設 0.005 = -46dB） */
	silenceThreshold?: number;
	/** Trim 保留邊緣 padding（預設 25ms，避免切到中文入聲/輕聲） */
	trimPadMs?: number;
	/** 組合句相鄰段重疊 ms（預設 30，蓋過接點避免「啪」斷裂感） */
	sentenceOverlapMs?: number;
}

export class PrerenderedTtsPlayer {
	private manifest: TtsManifest | null = null;
	private manifestPromise: Promise<TtsManifest | null> | null = null;
	private audioCtx: AudioContext | null = null;
	private audioUnlocked = false;
	private unlockSetup = false;
	private bufferCache = new Map<string, AudioBuffer>();
	private inflight = new Map<string, Promise<AudioBuffer | null>>();
	private currentSources: AudioBufferSourceNode[] = [];

	private readonly baseUrl: string;
	private readonly manifestUrl: string;
	private readonly enabledFn: () => boolean;
	private readonly playbackRate: number;
	private readonly silenceThreshold: number;
	private readonly trimPadMs: number;
	private readonly sentenceOverlapMs: number;

	constructor(opts: PrerenderedTtsOptions) {
		if (!opts.baseUrl) throw new Error("PrerenderedTtsPlayer: baseUrl required");
		this.baseUrl = opts.baseUrl.replace(/\/$/, "");
		this.manifestUrl = opts.manifestUrl ?? `${this.baseUrl}/manifest.json`;
		this.enabledFn = opts.enabled ?? (() => true);
		this.playbackRate = opts.playbackRate ?? 1.0;
		this.silenceThreshold = opts.silenceThreshold ?? 0.005;
		this.trimPadMs = opts.trimPadMs ?? 25;
		this.sentenceOverlapMs = opts.sentenceOverlapMs ?? 30;
	}

	/**
	 * 註冊 user gesture listener；第一次 click/touchend/keydown 解鎖 AudioContext。
	 * iOS Safari / 部分 Chrome 在無 user gesture 下 AudioContext.state="suspended"，
	 * source.start() 不會發聲。建議 app 啟動時呼叫一次。
	 */
	setupAudioUnlock(): void {
		if (typeof window === "undefined") return;
		if (this.unlockSetup) return;
		this.unlockSetup = true;

		const unlock = () => {
			if (this.audioUnlocked) return;
			const ctx = this.getCtx();
			if (!ctx) return;
			const done = () => {
				this.audioUnlocked = true;
			};
			if (ctx.state === "suspended") {
				ctx.resume()
					.then(done)
					.catch(() => {
						/* 失敗就算了，下次 gesture 再試 */
					});
			} else {
				done();
			}
		};

		const opts = { once: true, passive: true } as AddEventListenerOptions;
		window.addEventListener("click", unlock, opts);
		window.addEventListener("touchend", unlock, opts);
		window.addEventListener("keydown", unlock, opts);
	}

	isAudioUnlocked(): boolean {
		return this.audioUnlocked;
	}

	/** 第一次呼叫 fetch manifest，後續 cache */
	async loadManifest(): Promise<TtsManifest | null> {
		if (this.manifest) return this.manifest;
		if (this.manifestPromise) return this.manifestPromise;

		this.manifestPromise = (async () => {
			try {
				const resp = await fetch(this.manifestUrl, { cache: "force-cache" });
				if (!resp.ok) return null;
				const data = (await resp.json()) as TtsManifest;
				this.manifest = data;
				return data;
			} catch {
				return null;
			}
		})();
		return this.manifestPromise;
	}

	hasPhrase(key: string): boolean {
		return this.manifest?.phrases?.[key] !== undefined;
	}

	/** 取消目前正在播的所有 source */
	cancelAll(): void {
		for (const src of this.currentSources) {
			try {
				src.onended = null;
				src.stop();
			} catch {
				/* noop — 可能已 ended */
			}
		}
		this.currentSources = [];
	}

	/**
	 * 播一個 phrase。
	 * 回傳 true = 有播；false = 缺檔 / enabled=false / ctx 未解鎖
	 * Caller 拿 false 應自行 fallback（如 Web Speech API）。
	 */
	async playPhrase(key: string): Promise<boolean> {
		if (typeof window === "undefined") return false;
		if (!this.enabledFn()) return false;
		await this.loadManifest();
		if (!this.hasPhrase(key)) return false;
		const buf = await this.loadBuffer(key);
		if (!buf) return false;
		const ctx = this.getCtx();
		if (!ctx || ctx.state === "suspended") return false;
		return this.scheduleBuffers([buf]);
	}

	/**
	 * 連續播多個 phrase（組合句）。任一 key 缺檔 → 整句回 false（避免半句念對半句缺）。
	 *
	 * 例：playSentence(['num_zh_5', 'op_sub', 'num_zh_1', 'q_equals_what'])
	 *
	 * 段與段精準對接（Web Audio AudioBufferSourceNode.start(when)）+ 重疊 sentenceOverlapMs
	 * → 聽起來連續一句、無 cold start gap。
	 */
	async playSentence(keys: string[]): Promise<boolean> {
		if (typeof window === "undefined") return false;
		if (keys.length === 0) return false;
		if (!this.enabledFn()) return false;
		await this.loadManifest();
		for (const key of keys) {
			if (!this.hasPhrase(key)) return false;
		}
		const buffers = await Promise.all(keys.map((k) => this.loadBuffer(k)));
		if (buffers.some((b) => b === null)) return false;
		const ctx = this.getCtx();
		if (!ctx || ctx.state === "suspended") return false;
		return this.scheduleBuffers(buffers as AudioBuffer[]);
	}

	/**
	 * 數字 → manifest key（中文 0-100、英文 0-9）。
	 * 超出範圍回 null，caller 應 fallback。
	 */
	static numberKey(n: number, lang: "zh-TW" | "en-US"): string | null {
		if (!Number.isInteger(n)) return null;
		if (lang === "zh-TW") {
			if (n >= 0 && n <= 100) return `num_zh_${n}`;
			return null;
		}
		if (n >= 0 && n <= 9) return `num_en_${n}`;
		return null;
	}

	/** 運算符 → manifest key */
	static operationKey(op: string): string | null {
		switch (op) {
			case "+":
			case "addition":
				return "op_add";
			case "-":
			case "−":
			case "subtraction":
				return "op_sub";
			case "*":
			case "×":
			case "multiplication":
				return "op_mul";
			case "/":
			case "÷":
			case "division":
				return "op_div";
			default:
				return null;
		}
	}

	// -----------------------------------------------------------------------
	// Internal
	// -----------------------------------------------------------------------

	private getCtx(): AudioContext | null {
		if (typeof window === "undefined") return null;
		if (this.audioCtx) return this.audioCtx;
		type CtxCtor = typeof AudioContext;
		const w = window as Window & { webkitAudioContext?: CtxCtor };
		const Ctor: CtxCtor | undefined = window.AudioContext ?? w.webkitAudioContext;
		if (!Ctor) return null;
		this.audioCtx = new Ctor();
		return this.audioCtx;
	}

	/**
	 * decode 後 trim 首尾 silence。edge-tts MP3 每段前後有 200~500ms 靜音 padding，
	 * 串接播放時會聽到「字字停頓」。把 |sample| < silenceThreshold 的首尾剪掉、
	 * 留 trimPadMs 保護中文入聲/輕聲（amplitude 低，padding 太小會被切掉）。
	 */
	private trimSilence(ctx: AudioContext, buf: AudioBuffer): AudioBuffer {
		const numCh = buf.numberOfChannels;
		const len = buf.length;
		const data = buf.getChannelData(0); // edge-tts mono；stereo 兩聲道相同

		let start = 0;
		while (start < len && Math.abs(data[start]) < this.silenceThreshold) start++;
		let end = len - 1;
		while (end > start && Math.abs(data[end]) < this.silenceThreshold) end--;

		if (start >= end) return buf; // 整段都靜音？保險不剪
		const padSamples = Math.floor((this.trimPadMs / 1000) * buf.sampleRate);
		start = Math.max(0, start - padSamples);
		end = Math.min(len - 1, end + padSamples);

		const trimmedLen = end - start + 1;
		if (trimmedLen >= len - 16) return buf;
		const trimmed = ctx.createBuffer(numCh, trimmedLen, buf.sampleRate);
		for (let c = 0; c < numCh; c++) {
			const srcData = buf.getChannelData(c);
			const dstData = trimmed.getChannelData(c);
			for (let i = 0; i < trimmedLen; i++) {
				dstData[i] = srcData[start + i];
			}
		}
		return trimmed;
	}

	private loadBuffer(key: string): Promise<AudioBuffer | null> {
		const cached = this.bufferCache.get(key);
		if (cached) return Promise.resolve(cached);
		const pending = this.inflight.get(key);
		if (pending) return pending;

		const p = (async () => {
			const ctx = this.getCtx();
			if (!ctx) return null;
			try {
				const resp = await fetch(`${this.baseUrl}/${encodeURIComponent(key)}.mp3`, {
					cache: "force-cache",
				});
				if (!resp.ok) return null;
				const arr = await resp.arrayBuffer();
				const raw = await ctx.decodeAudioData(arr);
				const trimmed = this.trimSilence(ctx, raw);
				this.bufferCache.set(key, trimmed);
				return trimmed;
			} catch {
				return null;
			} finally {
				this.inflight.delete(key);
			}
		})();
		this.inflight.set(key, p);
		return p;
	}

	private scheduleBuffers(buffers: AudioBuffer[]): Promise<boolean> {
		return new Promise((resolve) => {
			const ctx = this.getCtx();
			if (!ctx) {
				resolve(false);
				return;
			}
			this.cancelAll();
			// 關 Web Speech 避免雙重發聲（若 caller 同時跑了 fallback）
			if (typeof window !== "undefined" && window.speechSynthesis) {
				try {
					window.speechSynthesis.cancel();
				} catch {
					/* noop */
				}
			}

			const startBase = ctx.currentTime + 0.02; // 留 20ms 給 scheduler
			let when = startBase;
			const sources: AudioBufferSourceNode[] = [];
			for (let i = 0; i < buffers.length; i++) {
				const buf = buffers[i];
				const src = ctx.createBufferSource();
				src.buffer = buf;
				src.playbackRate.value = this.playbackRate;
				src.connect(ctx.destination);
				try {
					src.start(when);
				} catch {
					for (const s of sources) {
						try {
							s.stop();
						} catch {
							/* noop */
						}
					}
					resolve(false);
					return;
				}
				sources.push(src);
				const dur = buf.duration / this.playbackRate;
				when += Math.max(0, dur - this.sentenceOverlapMs / 1000);
			}
			this.currentSources = sources;

			const last = sources[sources.length - 1];
			last.onended = () => {
				if (this.currentSources === sources) this.currentSources = [];
				resolve(true);
			};
		});
	}
}

// ---------------------------------------------------------------------------
// 範例：在 React 內使用
// ---------------------------------------------------------------------------
//
// import { useEffect, useRef } from 'react';
// import { PrerenderedTtsPlayer } from './prerendered-tts-player';
//
// const BASE = 'https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0';
//
// // 建議 module-level singleton（一個 app 一個 AudioContext）
// const ttsPlayer = new PrerenderedTtsPlayer({
//   baseUrl: `${BASE}/audio/tts-zh-tw`,
//   enabled: () => true,
// });
//
// export function useTtsPlayer() {
//   useEffect(() => {
//     ttsPlayer.setupAudioUnlock();
//     ttsPlayer.loadManifest();
//   }, []);
//   return ttsPlayer;
// }
//
// // 播放 "5 - 1 = ?"
// const tts = useTtsPlayer();
// async function speakProblem(a: number, op: '+'|'-'|'*'|'/', b: number) {
//   const ok = await tts.playSentence([
//     PrerenderedTtsPlayer.numberKey(a, 'zh-TW')!,
//     PrerenderedTtsPlayer.operationKey(op)!,
//     PrerenderedTtsPlayer.numberKey(b, 'zh-TW')!,
//     'q_equals_what',
//   ]);
//   if (!ok) {
//     // Fallback: Web Speech API
//     const u = new SpeechSynthesisUtterance(`${a} ${op} ${b} 等於多少`);
//     u.lang = 'zh-TW';
//     speechSynthesis.speak(u);
//   }
// }
