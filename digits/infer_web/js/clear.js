import canvas from "./canvas";
import { setChartData } from "./chart";
import { clearScalingCanvas } from "./scaling";

const clearButtonEl = document.querySelector('button.clear');

clearButtonEl.addEventListener('click', () => {
    canvas.clear();
    clearScalingCanvas();
    setChartData([0, 0, 0, 0, 0, 0, 0, 0, 0]);
});
