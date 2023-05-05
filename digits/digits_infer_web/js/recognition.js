import { recognize } from '../build/index';

import { drawingCanvasEl } from './canvas';
import { setChartData } from './chart';
import { scaleImage } from './scaling';

const IMAGE_SIZE = 28;

function prepareImage(imageData, targetSize) {
    let image = new Float64Array(targetSize ** 2);
    let pixelsCount = imageData.data.length / 4;

    if (pixelsCount != targetSize ** 2) {
        throw new Error('invalid image size');
    }

    for (let i = 0; i < pixelsCount; ++i) {
        let rgba = i * 4;

        // empty pixel is transparent black - rgba(0, 0, 0, 0)
        // non-empty pixel is opaque black  - rgba(0, 0, 0, 255)
        let alphaChannel = imageData.data[rgba + 3];

        // model was trained on images with f64 number pixels in range [-1, 1]
        image[i] = alphaChannel / 127.5 - 1;
    }

    return image;
}

export function recognizeAndUpdateChart() {
    const imageData = scaleImage(drawingCanvasEl, IMAGE_SIZE);
    const image = prepareImage(imageData, IMAGE_SIZE);

    const data = recognize(image);

    setChartData(data);
}