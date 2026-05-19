/**
 * LetterRecognizer — TF.js inference helper for letter classification (52 classes).
 *
 * 使用方式（給未來新專案 copy-paste）：
 *
 *   import { LetterRecognizer } from './letter-recognizer';
 *
 *   const recognizer = new LetterRecognizer();
 *   await recognizer.load(
 *     '/models/letter-v1/model.json',
 *     '/models/letter-v1/classes.json',
 *   );
 *
 *   // canvas 是 user 在 iPad 上手寫的 ROI，內含白底黑字筆畫
 *   const ctx = canvas.getContext('2d');
 *   const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
 *
 *   const topK = recognizer.predict(imageData, 3);
 *   // [{ label: 'A', confidence: 0.91 }, { label: 'a', confidence: 0.06 }, ...]
 *
 *   // 已知題目寫的是大寫 → 大小寫合併信心
 *   const caseInsensitive = recognizer.predictCaseInsensitive(imageData, 3);
 *
 * 訓練詳情：見 scripts/letter-training/README.md。
 * Model 規格：28×28×1 grayscale，內部統一黑底白字。
 */

// @ts-ignore — assumes @tensorflow/tfjs is installed in consumer project
import * as tf from "@tensorflow/tfjs";

export interface LetterClassesMeta {
	version: string;
	num_classes: number;
	expected_polarity: "black_bg_white_ink";
	input_shape: [number, number, number];
	labels: string[];
	case_pairs: number[][]; // [upper_idx, lower_idx] 共 26 對
	confusable_pairs: number[][]; // 11 對手寫長相幾乎一樣的字母
}

export interface LetterPrediction {
	label: string;
	confidence: number;
	classIdx: number;
}

export interface CaseInsensitivePrediction {
	/** "A/a" for confusable pairs, otherwise single letter */
	label: string;
	confidence: number;
	classIndices: number[];
}

export class LetterRecognizer {
	private model: tf.LayersModel | null = null;
	private meta: LetterClassesMeta | null = null;

	/** 載 model + classes.json。建議 app 啟動時 await 一次。 */
	async load(modelUrl: string, classesUrl: string): Promise<void> {
		const [model, metaResponse] = await Promise.all([
			tf.loadLayersModel(modelUrl),
			fetch(classesUrl),
		]);
		const meta = (await metaResponse.json()) as LetterClassesMeta;

		if (meta.expected_polarity !== "black_bg_white_ink") {
			throw new Error(
				`LetterRecognizer: expected_polarity=${meta.expected_polarity}, ` +
					`script 內部假設 black_bg_white_ink。重訓 model 或改 preprocessing。`,
			);
		}
		if (meta.num_classes !== meta.labels.length) {
			throw new Error("classes.json num_classes vs labels length mismatch");
		}

		this.model = model;
		this.meta = meta;
	}

	/** 釋放 model 記憶體（unmount 時呼叫） */
	dispose(): void {
		this.model?.dispose();
		this.model = null;
		this.meta = null;
	}

	/**
	 * Top-K 預測（label-aware，A 跟 a 是不同類）。
	 * @param imageData 任意尺寸 RGBA ImageData（白底黑字 0-255 — UI 常見 PadView 抓出來的格式）
	 * @param topK 預設 3
	 */
	predict(imageData: ImageData, topK = 3): LetterPrediction[] {
		const meta = this.assertReady();

		return tf.tidy(() => {
			const input = this.preprocess(imageData);
			const probsTensor = this.model!.predict(input) as tf.Tensor;
			const probs = Array.from(probsTensor.dataSync());

			return probs
				.map((confidence, classIdx) => ({
					label: meta.labels[classIdx],
					confidence,
					classIdx,
				}))
				.sort((a, b) => b.confidence - a.confidence)
				.slice(0, topK);
		});
	}

	/**
	 * Case-insensitive 預測：合併 26 個大小寫對的信心。
	 * 例如 c=0.4 + C=0.3 合併為 "C/c"=0.7。
	 *
	 * 用在「應用層已知題目大小寫」的場景（拼字遊戲 prompt 是 Apple → 第一格只接受 A or a）。
	 * 在 confusable_pairs（C/c, O/o, S/s 等手寫長相像的對）特別有用。
	 */
	predictCaseInsensitive(imageData: ImageData, topK = 3): CaseInsensitivePrediction[] {
		const meta = this.assertReady();

		return tf.tidy(() => {
			const input = this.preprocess(imageData);
			const probsTensor = this.model!.predict(input) as tf.Tensor;
			const probs = Array.from(probsTensor.dataSync());

			const mergedMap = new Map<string, CaseInsensitivePrediction>();
			for (const [upperIdx, lowerIdx] of meta.case_pairs) {
				const upper = meta.labels[upperIdx];
				const lower = meta.labels[lowerIdx];
				mergedMap.set(`${upper}/${lower}`, {
					label: `${upper}/${lower}`,
					confidence: probs[upperIdx] + probs[lowerIdx],
					classIndices: [upperIdx, lowerIdx],
				});
			}

			return Array.from(mergedMap.values())
				.sort((a, b) => b.confidence - a.confidence)
				.slice(0, topK);
		});
	}

	/** 公開 confusable_pairs 給應用層判斷要不要對 top-1 結果做 fallback */
	getConfusablePairs(): string[][] {
		const meta = this.assertReady();
		return meta.confusable_pairs.map(([upperIdx, lowerIdx]) => [
			meta.labels[upperIdx],
			meta.labels[lowerIdx],
		]);
	}

	private assertReady(): LetterClassesMeta {
		if (!this.model || !this.meta) {
			throw new Error("LetterRecognizer: call load() first");
		}
		return this.meta;
	}

	/**
	 * 白底黑字 ImageData → 28×28×1 黑底白字 normalize [0,1] tensor。
	 *
	 * 關鍵步驟（CLAUDE.md / digit pipeline 學到的）：
	 * 1. 任意尺寸 → 28×28（resize bilinear）
	 * 2. RGBA → grayscale（取 R channel 即可，灰階畫面三通道值相同）
	 * 3. 翻 polarity 白底黑字 → 黑底白字（training distribution）
	 * 4. normalize [0,1]
	 */
	private preprocess(imageData: ImageData): tf.Tensor4D {
		const [h, w, c] = this.meta!.input_shape;
		if (c !== 1) {
			throw new Error(`Expected 1-channel input, got ${c}`);
		}

		return tf.tidy(() => {
			// RGBA Uint8ClampedArray → tensor [H, W, 4]
			const rgba = tf.browser.fromPixels(imageData, 4);

			// Grayscale: 取 R channel
			const gray = rgba.slice([0, 0, 0], [-1, -1, 1]).cast("float32");

			// Resize 到 28×28
			const resized = tf.image.resizeBilinear(gray as tf.Tensor3D, [h, w]);

			// Polarity 翻：白底黑字 (255 - x) → 黑底白字
			const inverted = tf.sub(255.0, resized);

			// Normalize [0, 255] → [0, 1]
			const normalized = tf.div(inverted, 255.0);

			// 加 batch dim → [1, 28, 28, 1]
			return normalized.expandDims(0) as tf.Tensor4D;
		});
	}
}

// ---------------------------------------------------------------------------
// 範例：在 React 內使用
// ---------------------------------------------------------------------------
//
// import { useEffect, useRef, useState } from 'react';
// import { LetterRecognizer } from './letter-recognizer';
//
// export function useLetterRecognizer() {
//   const ref = useRef<LetterRecognizer | null>(null);
//   const [ready, setReady] = useState(false);
//
//   useEffect(() => {
//     const r = new LetterRecognizer();
//     r.load('/models/letter-v1/model.json', '/models/letter-v1/classes.json')
//       .then(() => { ref.current = r; setReady(true); });
//     return () => { ref.current?.dispose(); };
//   }, []);
//
//   return { recognizer: ref.current, ready };
// }
//
// // In component:
// const { recognizer, ready } = useLetterRecognizer();
// async function handleSubmit() {
//   if (!recognizer || !ready) return;
//   const imageData = canvasRef.current!.getContext('2d')!
//     .getImageData(0, 0, canvas.width, canvas.height);
//   const top = recognizer.predictCaseInsensitive(imageData, 3);
//   console.log(top); // [{ label: 'A/a', confidence: 0.93 }, ...]
// }
