/**
 * VoiceInputRecognizer — Web Speech Recognition API wrapper + 中文數字解析。
 *
 * 給「孩子說數字答題」的場景：用 webkitSpeechRecognition 把口說轉文字、
 * 再用 parseChineseNumber 把「十二」「二十三」「一百零五」轉成數字。
 *
 * 使用方式：
 *
 *   import { VoiceInputRecognizer, parseChineseNumber } from './voice-input-parser';
 *
 *   // 純解析（不開麥克風）
 *   parseChineseNumber('十二');     // 12
 *   parseChineseNumber('一百零五');  // 105
 *   parseChineseNumber('42');        // 42
 *
 *   // 開麥克風聽一次
 *   const recognizer = new VoiceInputRecognizer({ lang: 'zh-TW', timeoutMs: 5000 });
 *   if (!VoiceInputRecognizer.isSupported()) {
 *     // 瀏覽器不支援（Firefox / 部分桌面 Chrome）→ 走鍵盤輸入 fallback
 *   }
 *   const { transcript, number, confidence } = await recognizer.listen();
 *   // { transcript: '十二', number: 12, confidence: 0.93 }
 *
 * iPad Safari 支援 webkitSpeechRecognition。Firefox 桌面版不支援。
 */

const CHINESE_TO_NUM: Record<string, number> = {
	零: 0,
	一: 1,
	二: 2,
	三: 3,
	四: 4,
	五: 5,
	六: 6,
	七: 7,
	八: 8,
	九: 9,
	十: 10,
	兩: 2,
	百: 100,
	千: 1000,
};

/**
 * 解析中文 / 阿拉伯數字字串為阿拉伯數字。回 null 表示無法解析。
 *
 * 支援：
 *   '5'、'42'、'105'           純阿拉伯數字（含空白）
 *   '五'、'兩'                  單字數字
 *   '十二'、'二十三'             兩位數
 *   '一百零五'、'三百'           三位數
 *   '一千二百'                  四位數
 */
export function parseChineseNumber(text: string): number | null {
	const direct = text.replace(/\s/g, "");

	// 直接是阿拉伯數字
	if (/^\d+$/.test(direct)) {
		return Number.parseInt(direct, 10);
	}

	// 純單字
	if (CHINESE_TO_NUM[direct] !== undefined && direct !== "百" && direct !== "千") {
		return CHINESE_TO_NUM[direct];
	}

	// 複合中文數字（含位數單位）
	let result = 0;
	let current = 0;
	for (const char of direct) {
		const val = CHINESE_TO_NUM[char];
		if (val === undefined) continue;
		if (char === "十") {
			if (current === 0) current = 1;
			result += current * 10;
			current = 0;
		} else if (char === "百") {
			if (current === 0) current = 1;
			result += current * 100;
			current = 0;
		} else if (char === "千") {
			if (current === 0) current = 1;
			result += current * 1000;
			current = 0;
		} else {
			current = val;
		}
	}
	result += current;
	return result > 0 || direct === "零" ? result : null;
}

export interface VoiceResult {
	transcript: string;
	number: number | null;
	confidence: number;
}

export interface VoiceInputOptions {
	/** 辨識語言（預設 zh-TW） */
	lang?: string;
	/** 超時 ms（預設 5000） */
	timeoutMs?: number;
	/** maxAlternatives（預設 3，嘗試多個候選看哪個能解析成數字） */
	maxAlternatives?: number;
}

export class VoiceInputRecognizer {
	private readonly lang: string;
	private readonly timeoutMs: number;
	private readonly maxAlternatives: number;

	constructor(opts: VoiceInputOptions = {}) {
		this.lang = opts.lang ?? "zh-TW";
		this.timeoutMs = opts.timeoutMs ?? 5000;
		this.maxAlternatives = opts.maxAlternatives ?? 3;
	}

	/** 檢查瀏覽器是否支援 Web Speech Recognition API */
	static isSupported(): boolean {
		if (typeof window === "undefined") return false;
		const w = window as unknown as Record<string, unknown>;
		return !!(w.SpeechRecognition || w.webkitSpeechRecognition);
	}

	/**
	 * 開麥克風聽一次，回傳辨識結果。
	 * - 超時或 no-speech → resolve({ transcript: '', number: null, confidence: 0 })
	 * - 嘗試 maxAlternatives 個候選，第一個能解析成數字的回傳
	 * - 全部都無法解析 → 回傳 top-1 transcript 但 number=null
	 */
	listen(): Promise<VoiceResult> {
		return new Promise((resolve, reject) => {
			if (!VoiceInputRecognizer.isSupported()) {
				reject(new Error("Speech recognition not supported"));
				return;
			}

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const W = window as any;
			const SpeechRecognitionCtor = W.SpeechRecognition || W.webkitSpeechRecognition;
			const recognition = new SpeechRecognitionCtor();

			recognition.lang = this.lang;
			recognition.interimResults = false;
			recognition.maxAlternatives = this.maxAlternatives;
			recognition.continuous = false;

			const timeout = setTimeout(() => {
				try {
					recognition.stop();
				} catch {
					/* noop */
				}
				resolve({ transcript: "", number: null, confidence: 0 });
			}, this.timeoutMs);

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			recognition.onresult = (event: any) => {
				clearTimeout(timeout);
				const result = event.results?.[0];
				if (!result) {
					resolve({ transcript: "", number: null, confidence: 0 });
					return;
				}
				for (let i = 0; i < result.length; i++) {
					const alt = result[i];
					const num = parseChineseNumber(alt.transcript);
					if (num !== null) {
						resolve({
							transcript: alt.transcript,
							number: num,
							confidence: alt.confidence,
						});
						return;
					}
				}
				resolve({
					transcript: result[0].transcript,
					number: null,
					confidence: result[0].confidence,
				});
			};

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			recognition.onerror = (event: any) => {
				clearTimeout(timeout);
				if (event.error === "no-speech" || event.error === "aborted") {
					resolve({ transcript: "", number: null, confidence: 0 });
				} else {
					reject(new Error(`Speech recognition error: ${event.error}`));
				}
			};

			recognition.onend = () => {
				clearTimeout(timeout);
			};

			recognition.start();
		});
	}
}

// ---------------------------------------------------------------------------
// 範例：在 React 內使用
// ---------------------------------------------------------------------------
//
// import { useState } from 'react';
// import { VoiceInputRecognizer } from './voice-input-parser';
//
// export function VoiceAnswer({ expected }: { expected: number }) {
//   const [listening, setListening] = useState(false);
//   const [result, setResult] = useState<string>('');
//
//   if (!VoiceInputRecognizer.isSupported()) {
//     return <p>此瀏覽器不支援語音輸入，請改用鍵盤</p>;
//   }
//
//   async function handleClick() {
//     setListening(true);
//     const recognizer = new VoiceInputRecognizer({ lang: 'zh-TW' });
//     try {
//       const { transcript, number } = await recognizer.listen();
//       if (number === expected) setResult('答對了！');
//       else if (number !== null) setResult(`你說 ${transcript} (${number})，答案是 ${expected}`);
//       else setResult(`聽不清楚（${transcript}）— 再說一次？`);
//     } finally {
//       setListening(false);
//     }
//   }
//
//   return (
//     <button onClick={handleClick} disabled={listening}>
//       {listening ? '聽中…' : '按住說數字'}
//     </button>
//   );
// }
