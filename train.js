import "https://unpkg.com/ml5@1/dist/ml5.min.js";

ml5.setBackend("webgl");
const options = {
    task: 'classification',
    debug: true,
    layers: [
        { type: 'dense', units: 32, activation: 'relu' },
        { type: 'dense', units: 32, activation: 'relu' },
        { type: 'dense', units: 32, activation: 'relu' },
        { type: 'dense', activation: 'softmax' },
    ]
};
const nn = ml5.neuralNetwork(options);

const trainButton = document.getElementById("trainButton");

trainButton.addEventListener("click", fetchTrainingData);

async function fetchTrainingData() {
    fetch("TrainingData.json")
        .then((response) => response.json())
        .then((data) => {
            trainNN(data);
            console.log(data);
        });
}

async function trainNN(trainingData) {
    for (let label in trainingData) {
        trainingData[label].forEach(data => {
            nn.addData(data, { label: label });
        });
    }

    nn.normalizeData();
    nn.train({ epochs: 200 }, () => finishedTraining());

    function finishedTraining() {
        console.log("Training complete!");
        nn.save("model", () => console.log("Model was saved!"));
    }
}
