import Chart from 'chart.js/auto';
import ChartDataLabels from 'chartjs-plugin-datalabels';

const chartCanvasEl = document.querySelector('#chart');

function chartConfigBuilder(chartEl) {
    Chart.register(ChartDataLabels);
    return new Chart(chartEl, {
        plugins: [ChartDataLabels],
        type: "bar",
        data: {
            labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            datasets: [
                {
                    data: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    borderWidth: 0,
                    fill: true,
                    backgroundColor: "#247ABF",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: true,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: true,
                },
                datalabels: {
                    color: "white",
                    formatter: function (value, context) {
                        return value.toFixed(2);
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                },
            },
        },
    });
}

const chart = chartConfigBuilder(chartCanvasEl);

export function setChartData(data) {
    chart.data.datasets[0].data = data;
    chart.update();
}
