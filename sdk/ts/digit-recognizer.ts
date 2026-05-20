/**
 * DigitRecognizer — TF.js inference helper for handwritten digit recognition (0-9).
 *
 * 使用方式（給未來新專案 copy-paste）：
 *
 *   import { DigitRecognizer } from './digit-recognizer';
 *
 *   const recognizer = new DigitRecognizer({
 *     modelUrl: 'https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0/models/digit-v1/model.json',
 *     cacheKey: 'indexeddb://my-app-digit-v1',  // 可選；不傳就不 cache
 *   });
 *   await recognizer.load();
 *
 *   // 單字符（建議路徑：每格 PadView 抓 200×200 白底黑字 dataURL）
 *   const result = await recognizer.predictFromDataUrl(dataUrl, { useTta: true });
 *   // { digit: 7, confidence: 0.93 } 或 { digit: null, confidence: 0.4 }（信心不足）
 *
 *   // 多位數（一張 canvas 含「12」「345」之類）
 *   const multi = await recognizer.predictMulti(canvas);
 *   // { text: '12', confidence: 0.88 }
 *
 *   recognizer.dispose();  // unmount 時釋放 model 記憶體
 *
 * Model 規格：v1 是 28×28×1 grayscale，內部訓練 black_bg_white_ink。
 * Input 預設假設 white_bg_black_ink（手寫常見），SDK 內部 invert polarity。
 *
 * 為何需要 centerByMassPreprocess（內建、預設開）：
 *   MNIST 訓練資料字佔 20×20、重心對齊 28×28 的 (14,14)。
 *   App 端的 200×200 input 直接 resizeBilinear 28×28 會讓字實際只佔 ~11×11，
 *   嚴重 domain mismatch。包進 SDK 後 baseline 60-75% → 75-82%。
 *
 * 訓練詳情：見 scripts/train-digit/。
 */

// @ts-ignore — assumes @tensorflow/tfjs is installed in consumer project
import * as tf from "@tensorflow/tfjs";

export interface DigitPrediction {
	digit: number | null;
	confidence: number;
}

export interface DigitMultiResult {
	text: string;
	confidence: number;
}

export interface DigitRecognizerOptions {
	/** Model.json URL（CDN 或 same-origin 路徑） */
	modelUrl: string;
	/** IndexedDB cache key（如 "indexeddb://my-app-digit-v1"）；不傳 = 不快取 */
	cacheKey?: string;
	/** predict 信心低於此值視為 null（預設 0.6） */
	confidenceThreshold?: number;
	/** Input ImageData 的 polarity（預設 white_bg_black_ink，SDK 內 invert） */
	expectedPolarity?: "white_bg_black_ink" | "black_bg_white_ink";
}

export interface DigitPredictOptions {
	/** 跑 test-time augmentation（±4°/±8° rotation soft voting）— 預設 false */
	useTta?: boolean;
	/** 跳過 centerByMassPreprocess（input 已 MNIST-aligned 才關） */
	skipCentering?: boolean;
}

export class DigitRecognizer {
	private model: tf.LayersModel | null = null;
	private modelLoading: Promise<void> | null = null;
	private warnedCpuBackend = false;

	constructor(private opts: DigitRecognizerOptions) {
		if (!opts.modelUrl) throw new Error("DigitRecognizer: modelUrl required");
	}

	/** 載 model（先試 IndexedDB cache、miss 再 fetch + 存 cache）。app 啟動時 await 一次。 */
	async load(): Promise<void> {
		if (this.model) return;
		if (this.modelLoading) return this.modelLoading;

		this.modelLoading = (async () => {
			const { modelUrl, cacheKey } = this.opts;

			if (cacheKey) {
				try {
					this.model = await tf.loadLayersModel(cacheKey);
					return;
				} catch {
					// cache miss → fetch
				}
			}

			this.model = await tf.loadLayersModel(modelUrl);
			if (cacheKey) {
				try {
					await this.model.save(cacheKey);
				} catch {
					// 存 cache 失敗不阻擋（私密模式 / Storage quota 滿）
				}
			}
		})();
		return this.modelLoading;
	}

	/** 釋放 model 記憶體（unmount 時呼叫） */
	dispose(): void {
		this.model?.dispose();
		this.model = null;
		this.modelLoading = null;
	}

	/**
	 * 單字符預測。
	 *
	 * @param imageData 任意尺寸 RGBA ImageData（預設假設白底黑字筆畫）
	 * @param opts useTta=true 跑 ±4°/±8° TTA；skipCentering=true 跳過重心置中
	 */
	async predict(
		imageData: ImageData,
		opts: DigitPredictOptions = {},
	): Promise<DigitPrediction> {
		this.assertReady();
		this.warnIfCpuBackend();

		const centered = opts.skipCentering
			? imageData
			: centerByMassPreprocess(imageData);
		if (!centered) return { digit: null, confidence: 0 };

		if (opts.useTta) return this.predictWithTta(centered);
		return this.predictSingle(centered);
	}

	/**
	 * 從 dataURL 預測（PadView 常見格式）。內部解 dataURL → ImageData 後走 predict()。
	 */
	async predictFromDataUrl(
		dataUrl: string,
		opts: DigitPredictOptions = {},
	): Promise<DigitPrediction> {
		if (typeof document === "undefined") return { digit: null, confidence: 0 };
		const imageData = await dataUrlToImageData(dataUrl);
		if (!imageData) return { digit: null, confidence: 0 };
		return this.predict(imageData, opts);
	}

	/**
	 * 多位數預測：canvas 內含「12」「345」等多字。
	 * 流程：垂直投影找字界 → 逐一切出單字 region → predictSingle → 串接。
	 */
	async predictMulti(canvas: HTMLCanvasElement): Promise<DigitMultiResult> {
		this.assertReady();
		const ctx = canvas.getContext("2d");
		if (!ctx) return { text: "", confidence: 0 };

		const full = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const hasInk = Array.from(full.data).some((v, i) => i % 4 !== 3 && v < 200);
		if (!hasInk) return { text: "", confidence: 0 };

		const regions = findDigitBounds(full);
		if (regions.length === 0) return { text: "", confidence: 0 };

		const digits: string[] = [];
		let totalConf = 0;
		const threshold = this.opts.confidenceThreshold ?? 0.6;

		for (const r of regions) {
			const region = extractRegion(canvas, r.x, r.width);
			const centered = centerByMassPreprocess(region);
			if (!centered) continue;
			const { digit, confidence } = await this.predictSingle(centered);
			if (digit !== null && confidence > threshold * 0.5) {
				// multi 路徑門檻較鬆（單字邊界可能切不乾淨，0.3 比 0.6 適合）
				digits.push(String(digit));
				totalConf += confidence;
			}
		}

		if (digits.length === 0) return { text: "", confidence: 0 };
		return { text: digits.join(""), confidence: totalConf / digits.length };
	}

	private async predictSingle(imageData: ImageData): Promise<DigitPrediction> {
		const input = this.preprocess(imageData);
		const pred = this.model!.predict(input) as tf.Tensor;
		const probs = await pred.data();
		input.dispose();
		pred.dispose();

		let maxProb = 0;
		let maxDigit = 0;
		for (let i = 0; i < probs.length; i++) {
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				maxDigit = i;
			}
		}
		const threshold = this.opts.confidenceThreshold ?? 0.6;
		if (maxProb < threshold) return { digit: null, confidence: maxProb };
		return { digit: maxDigit, confidence: maxProb };
	}

	/**
	 * Test-time augmentation soft voting。
	 *   1. Base predict 一次，max prob ≥ 0.9 直接回（省 ~70% 算力）
	 *   2. 否則跑 ±4°/±8° + 原圖 共 5 個 view 一起 predict，softmax 平均
	 *   3. max prob < 0.6 視為失敗
	 *
	 * 不做 shift augmentation：centerByMassPreprocess 已重心置中。
	 */
	private async predictWithTta(imageData: ImageData): Promise<DigitPrediction> {
		const base = await this.predictSingle(imageData);
		if (base.confidence >= 0.9 && base.digit !== null) return base;

		const avg = tf.tidy(() => {
			const input = this.preprocess(imageData);
			const radians = [-8, -4, 0, 4, 8].map((d) => (d * Math.PI) / 180);
			// fillValue=0：input 已 invert 為 black-bg white-ink，rotation 邊緣填黑色才不污染
			const views = radians.map((r) =>
				r === 0 ? input : tf.image.rotateWithOffset(input, r, 0, [0.5, 0.5]),
			);
			const batched = tf.concat(views, 0);
			const pred = this.model!.predict(batched) as tf.Tensor;
			return pred.mean(0);
		});
		const avgProbs = await avg.data();
		avg.dispose();

		let maxProb = 0;
		let maxDigit = 0;
		for (let i = 0; i < avgProbs.length; i++) {
			if (avgProbs[i] > maxProb) {
				maxProb = avgProbs[i];
				maxDigit = i;
			}
		}
		const threshold = this.opts.confidenceThreshold ?? 0.6;
		if (maxProb < threshold) return { digit: null, confidence: maxProb };
		return { digit: maxDigit, confidence: maxProb };
	}

	/**
	 * RGBA ImageData → 28×28×1 normalized [0,1] tensor。
	 * 內部根據 expectedPolarity 決定要不要 invert。
	 */
	private preprocess(imageData: ImageData): tf.Tensor4D {
		return tf.tidy(() => {
			let tensor = tf.browser.fromPixels(imageData, 1).toFloat();
			const polarity = this.opts.expectedPolarity ?? "white_bg_black_ink";
			if (polarity === "white_bg_black_ink") {
				// Model 訓的是 black_bg_white_ink，invert
				tensor = tf.sub(255, tensor);
			}
			const resized = tf.image.resizeBilinear(tensor as tf.Tensor3D, [28, 28]);
			const normalized = tf.div(resized, 255);
			return normalized.expandDims(0) as tf.Tensor4D;
		});
	}

	private assertReady(): void {
		if (!this.model) throw new Error("DigitRecognizer: call load() first");
	}

	private warnIfCpuBackend(): void {
		if (this.warnedCpuBackend) return;
		if (tf.getBackend() === "cpu") {
			this.warnedCpuBackend = true;
			console.warn(
				"[DigitRecognizer] backend=cpu，推論延遲會拉長 5-10×（iPad Safari WebGL 失效或不可用）",
			);
		}
	}
}

// ---------------------------------------------------------------------------
// Util: MNIST-style 重心置中
// ---------------------------------------------------------------------------

/**
 * 任意尺寸白底黑字 ImageData → 28×28 重心對齊 (14,14) 的 ImageData。
 * 沒 ink 回 null（呼叫端視為留空）。
 *
 * 流程：
 *   1. threshold gray < 200 找 ink bbox
 *   2. bbox 等比 fit 到 20×20
 *   3. intensity-weighted 重心
 *   4. 放進 28×28 重心對齊 (14,14)
 */
export function centerByMassPreprocess(imageData: ImageData): ImageData | null {
	if (typeof document === "undefined") return null;
	const { data, width, height } = imageData;
	const threshold = 200;

	let minX = width;
	let minY = height;
	let maxX = -1;
	let maxY = -1;
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			const idx = (y * width + x) * 4;
			const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
			if (gray < threshold) {
				if (x < minX) minX = x;
				if (x > maxX) maxX = x;
				if (y < minY) minY = y;
				if (y > maxY) maxY = y;
			}
		}
	}
	if (maxX < 0) return null;

	const bboxW = maxX - minX + 1;
	const bboxH = maxY - minY + 1;
	const fitSize = 20;
	const scale = Math.min(fitSize / bboxW, fitSize / bboxH);
	const scaledW = Math.max(1, Math.round(bboxW * scale));
	const scaledH = Math.max(1, Math.round(bboxH * scale));

	const srcCanvas = document.createElement("canvas");
	srcCanvas.width = width;
	srcCanvas.height = height;
	const srcCtx = srcCanvas.getContext("2d");
	if (!srcCtx) return null;
	srcCtx.putImageData(imageData, 0, 0);

	const tmpCanvas = document.createElement("canvas");
	tmpCanvas.width = scaledW;
	tmpCanvas.height = scaledH;
	const tmpCtx = tmpCanvas.getContext("2d");
	if (!tmpCtx) return null;
	tmpCtx.fillStyle = "#ffffff";
	tmpCtx.fillRect(0, 0, scaledW, scaledH);
	tmpCtx.drawImage(srcCanvas, minX, minY, bboxW, bboxH, 0, 0, scaledW, scaledH);

	const scaledData = tmpCtx.getImageData(0, 0, scaledW, scaledH).data;
	let sumX = 0;
	let sumY = 0;
	let totalWeight = 0;
	for (let y = 0; y < scaledH; y += 1) {
		for (let x = 0; x < scaledW; x += 1) {
			const idx = (y * scaledW + x) * 4;
			const gray = (scaledData[idx] + scaledData[idx + 1] + scaledData[idx + 2]) / 3;
			const w = Math.max(0, 255 - gray);
			sumX += x * w;
			sumY += y * w;
			totalWeight += w;
		}
	}
	if (totalWeight === 0) return null;
	const centroidX = sumX / totalWeight;
	const centroidY = sumY / totalWeight;

	const dstSize = 28;
	const dstCanvas = document.createElement("canvas");
	dstCanvas.width = dstSize;
	dstCanvas.height = dstSize;
	const dstCtx = dstCanvas.getContext("2d");
	if (!dstCtx) return null;
	dstCtx.fillStyle = "#ffffff";
	dstCtx.fillRect(0, 0, dstSize, dstSize);

	const dx = Math.round(dstSize / 2 - centroidX);
	const dy = Math.round(dstSize / 2 - centroidY);
	dstCtx.drawImage(tmpCanvas, dx, dy);

	return dstCtx.getImageData(0, 0, dstSize, dstSize);
}

// ---------------------------------------------------------------------------
// Internal: 多位數切割 helpers
// ---------------------------------------------------------------------------

function findDigitBounds(imageData: ImageData): Array<{ x: number; width: number }> {
	const { data, width, height } = imageData;
	const threshold = 200;
	const verticalProjection = new Array(width).fill(0);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = (y * width + x) * 4;
			const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
			if (gray < threshold) verticalProjection[x]++;
		}
	}

	const regions: Array<{ x: number; width: number }> = [];
	let inRegion = false;
	let regionStart = 0;
	const minGap = Math.max(3, Math.floor(width * 0.02));
	let gapCount = 0;

	for (let x = 0; x < width; x++) {
		if (verticalProjection[x] > 0) {
			if (!inRegion) {
				regionStart = x;
				inRegion = true;
			}
			gapCount = 0;
		} else if (inRegion) {
			gapCount++;
			if (gapCount > minGap) {
				const regionEnd = x - gapCount;
				const regionWidth = regionEnd - regionStart + 1;
				if (regionWidth > 5) regions.push({ x: regionStart, width: regionWidth });
				inRegion = false;
				gapCount = 0;
			}
		}
	}
	if (inRegion) {
		const regionWidth = width - regionStart;
		if (regionWidth > 5) regions.push({ x: regionStart, width: regionWidth });
	}
	return regions;
}

function extractRegion(
	canvas: HTMLCanvasElement,
	x: number,
	regionWidth: number,
): ImageData {
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("canvas context unavailable");

	const fullData = ctx.getImageData(x, 0, regionWidth, canvas.height);
	let top = canvas.height;
	let bottom = 0;
	const threshold = 200;

	for (let row = 0; row < canvas.height; row++) {
		for (let col = 0; col < regionWidth; col++) {
			const idx = (row * regionWidth + col) * 4;
			const gray = (fullData.data[idx] + fullData.data[idx + 1] + fullData.data[idx + 2]) / 3;
			if (gray < threshold) {
				top = Math.min(top, row);
				bottom = Math.max(bottom, row);
			}
		}
	}

	const padding = Math.max(10, Math.floor(regionWidth * 0.15));
	const cropTop = Math.max(0, top - padding);
	const cropBottom = Math.min(canvas.height, bottom + padding);
	const cropLeft = Math.max(0, x - padding);
	const cropWidth = regionWidth + padding * 2;
	const cropHeight = cropBottom - cropTop;

	const size = Math.max(cropWidth, cropHeight);
	const squareCanvas = document.createElement("canvas");
	squareCanvas.width = size;
	squareCanvas.height = size;
	const squareCtx = squareCanvas.getContext("2d");
	if (!squareCtx) throw new Error("square canvas context unavailable");
	squareCtx.fillStyle = "#ffffff";
	squareCtx.fillRect(0, 0, size, size);
	const offsetX = Math.floor((size - cropWidth) / 2);
	const offsetY = Math.floor((size - cropHeight) / 2);
	squareCtx.drawImage(
		canvas,
		cropLeft,
		cropTop,
		cropWidth,
		cropHeight,
		offsetX,
		offsetY,
		cropWidth,
		cropHeight,
	);
	return squareCtx.getImageData(0, 0, size, size);
}

async function dataUrlToImageData(dataUrl: string): Promise<ImageData | null> {
	return new Promise((resolve) => {
		const img = new Image();
		img.onload = () => {
			const canvas = document.createElement("canvas");
			canvas.width = img.width;
			canvas.height = img.height;
			const ctx = canvas.getContext("2d");
			if (!ctx) return resolve(null);
			ctx.drawImage(img, 0, 0);
			resolve(ctx.getImageData(0, 0, img.width, img.height));
		};
		img.onerror = () => resolve(null);
		img.src = dataUrl;
	});
}

// ---------------------------------------------------------------------------
// 範例：在 React 內使用
// ---------------------------------------------------------------------------
//
// import { useEffect, useRef, useState } from 'react';
// import { DigitRecognizer } from './digit-recognizer';
//
// const BASE = 'https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0';
//
// export function useDigitRecognizer() {
//   const ref = useRef<DigitRecognizer | null>(null);
//   const [ready, setReady] = useState(false);
//
//   useEffect(() => {
//     const r = new DigitRecognizer({
//       modelUrl: `${BASE}/models/digit-v1/model.json`,
//       cacheKey: 'indexeddb://my-app-digit-v1',
//     });
//     r.load().then(() => {
//       ref.current = r;
//       setReady(true);
//     });
//     return () => { ref.current?.dispose(); };
//   }, []);
//
//   return { recognizer: ref.current, ready };
// }
//
// // In component:
// const { recognizer, ready } = useDigitRecognizer();
// async function handleSubmit(dataUrl: string) {
//   if (!recognizer || !ready) return;
//   const { digit, confidence } = await recognizer.predictFromDataUrl(dataUrl, { useTta: true });
//   console.log(digit, confidence);
// }
