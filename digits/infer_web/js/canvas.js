import fabric from 'fabric';

const canvas = new fabric.fabric.Canvas("drawing", { isDrawingMode: true });
canvas.freeDrawingBrush.width = 50;

export default canvas;

export const drawingCanvasWrapper = document.querySelector(".drawing-canvas-wrapper");
export const drawingCanvasContainerEl = document.querySelector('.canvas-container');
export const drawingCanvasEl = document.querySelector('canvas#drawing');

function resizeCanvas() {
    const rect = drawingCanvasWrapper.getBoundingClientRect();

    const size = Math.min(rect.width, rect.height);

    canvas.setWidth(size);
    canvas.setHeight(size);
    canvas.renderAll();
};

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

