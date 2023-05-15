import { infer_digit } from '../build/index';

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

        // empty pixel is opaque white     - rgba(255, 255, 255, 255)
        // non-empty pixel is opaque black - rgba(0, 0, 0, 255)
        let ch = 255 - imageData.data[rgba];

        // convert image pixels from [0, 255] to [-1, 1]
        image[i] = ch / 127.5 - 1;
    }

    return image;
}

export function recognizeAndUpdateChart() {
    const imageData = scaleImage(drawingCanvasEl, IMAGE_SIZE);
    const image = prepareImage(imageData, IMAGE_SIZE);

    const data = infer_digit(image);

    setChartData(data);
}