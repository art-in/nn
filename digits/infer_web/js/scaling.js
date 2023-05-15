const scalingCanvasEl = document.querySelector('canvas.scaling');
const scalingCtx = scalingCanvasEl.getContext('2d', { willReadFrequently: true });

scalingCtx.imageSmoothingEnabled = true;

export function scaleImage(sourceCanvas, targetSize) {
    const sourceImageBoundaries = getSquareImageBoundariesWithPadding(sourceCanvas);

    scalingCtx.clearRect(0, 0, targetSize, targetSize);
    scalingCtx.drawImage(
        sourceCanvas,
        sourceImageBoundaries.x, sourceImageBoundaries.y,
        sourceImageBoundaries.width, sourceImageBoundaries.height,
        0, 0,
        targetSize, targetSize
    );

    return scalingCtx.getImageData(0, 0, targetSize, targetSize);
}

export function clearScalingCanvas() {
    scalingCtx.clearRect(0, 0, scalingCanvasEl.width, scalingCanvasEl.height);
}


// finds bounding square subregion inside canvas that contains entire drawn image with padding.
// it allows to focus on drawn image and eliminate empty areas
function getSquareImageBoundariesWithPadding(canvas) {
    const ctx = canvas.getContext('2d');
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // find bounding rectangle
    let leftmostX = imageData.width - 1;
    let rightmostX = 0;

    let topmostY = imageData.height - 1;
    let bottommostY = 0;

    let pixelsCount = imageData.data.length / 4;

    for (let pixelIdx = 0; pixelIdx < pixelsCount; pixelIdx++) {
        const channelIdx = pixelIdx * 4;

        const x = pixelIdx % imageData.width;
        const y = pixelIdx / imageData.width;

        if (imageData.data[channelIdx + 3] !== 0) {
            leftmostX = Math.min(leftmostX, x);
            rightmostX = Math.max(rightmostX, x);

            topmostY = Math.min(topmostY, y);
            bottommostY = Math.max(bottommostY, y);
        }
    }

    let width = rightmostX - leftmostX;
    let height = bottommostY - topmostY;

    // reshape found rectangle to square
    const paddingRatio = 1.1;
    const squareEdgeSize = Math.max(width, height) * paddingRatio;

    const squaredDiffX = squareEdgeSize - width;
    const squaredX = leftmostX - squaredDiffX / 2;

    const squaredDiffY = squareEdgeSize - height;
    const squaredY = topmostY - squaredDiffY / 2;

    return {
        x: squaredX,
        y: squaredY,
        width: squareEdgeSize,
        height: squareEdgeSize
    };
}
