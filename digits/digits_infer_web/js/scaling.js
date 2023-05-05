const scalingCanvasEl = document.querySelector('canvas.scaling');
const scalingCtx = scalingCanvasEl.getContext('2d', { willReadFrequently: true });

scalingCtx.imageSmoothingEnabled = true;

export function scaleImage(canvas, targetSize) {
    const originalSize = canvas.width;
    const scaleRatio = targetSize / originalSize;

    scalingCtx.clearRect(0, 0, targetSize, targetSize);

    scalingCtx.save();
    scalingCtx.scale(scaleRatio, scaleRatio);
    scalingCtx.drawImage(canvas, 0, 0);
    scalingCtx.restore();

    const res = scalingCtx.getImageData(0, 0, targetSize, targetSize);

    return res;
}

export function clearScalingCanvas() {
    scalingCtx.clearRect(0, 0, scalingCanvasEl.width, scalingCanvasEl.height);
}