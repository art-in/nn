import lodash from 'lodash';

import canvas, { drawingCanvasContainerEl } from "./canvas";
import { recognizeAndUpdateChart } from "./recognition";

function onDrawingStep() {
    canvas.freeDrawingBrush._finalizeAndAddPath();
    recognizeAndUpdateChart();
}

const onDrawingStepDebounced = lodash.debounce(
    onDrawingStep,
    1000,
    { leading: false, trailing: true }
);

let isDrawing = false;

drawingCanvasContainerEl.addEventListener('pointerdown', () => {
    isDrawing = true;
});

drawingCanvasContainerEl.addEventListener('pointerup', () => {
    isDrawing = false;
    onDrawingStepDebounced();
});
drawingCanvasContainerEl.addEventListener('pointermove', () => {
    if (isDrawing) {
        onDrawingStepDebounced();
    }
});

