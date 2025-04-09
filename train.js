ml5.setBackend("webgl");
const options = {
    task: 'classification',
    debug: true,
    layers: [
        {type: 'dense', units: 128, activation: 'relu'},
        {type: 'dense', units: 256, activation: 'relu'},
        {type: 'dense', units: 64, activation: 'relu'},
        {type: 'dense', activation: 'softmax'},
    ],
    learningRate: 0.01,
    epochs: 100
};
const nn = ml5.neuralNetwork(options);

let testData = [];

const trainButton = document.getElementById("trainButton");
const testButton = document.getElementById("testButton");

trainButton.addEventListener("click", fetchTrainingData);
testButton.addEventListener("click", testModel);

async function fetchTrainingData() {
    fetch("TrainingData.json")
        .then((response) => response.json())
        .then((data) => {
            const flatData = [];
            for (let label in data) {
                data[label].forEach(poseData => {
                    flatData.push({pose: poseData, label: label});
                });
            }
            flatData.sort(() => Math.random() - 0.5);

            const splitIndex = Math.floor(flatData.length * 0.8);
            const trainData = flatData.slice(0, splitIndex);
            testData = flatData.slice(splitIndex);

            trainNN(trainData);
        });
}

async function trainNN(trainingData) {
    trainingData.forEach(item => {
        nn.addData(item.pose, {label: item.label});
    });

    nn.normalizeData();
    nn.train({ epochs: options.epochs }, () => finishedTraining());

    function finishedTraining() {
        console.log("Training complete!");
        nn.save("model", () => console.log("Model was saved!"));
    }
}

async function testModel() {
    let correct = 0;

    if (testData.length === 0) {
        console.error("No test data available");
        return;
    }

    for (const testItem of testData) {
        const prediction = await nn.classify(testItem.pose);
        if (prediction[0].label === testItem.label) {
            correct++;
        }
    }

    const accuracy = (correct / testData.length * 100).toFixed(2);
    console.log(`Accuracy: ${accuracy}%`);

    const accuracyDisplay = document.getElementById("accuracyDisplay");
    if (accuracyDisplay) {
        accuracyDisplay.textContent = `Accuracy: ${accuracy}%`;
    }
}
