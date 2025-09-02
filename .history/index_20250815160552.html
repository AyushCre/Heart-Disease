<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body class="bg-gray-50 text-gray-800">

<div class="container mx-auto p-4 sm:p-6 lg:p-8 max-w-4xl">
    
    <header class="text-center mb-8">
        <h1 class="text-3xl sm:text-4xl font-bold text-gray-900">Heart Disease Prediction System</h1>
        <p class="mt-2 text-md text-gray-600">Fill in the details to get a quick prediction.</p>
    </header>

    <!-- form -->
    <main class="bg-white p-6 sm:p-8 rounded-2xl shadow-lg">
        <form id="prediction-form" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            
            <div>
                <label for="age" class="block text-sm font-medium">Age</label>
                <input type="number" id="age" name="age" min="1" max="120" required
                    class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="e.g., 52">
            </div>

            <div>
                <label for="sex" class="block text-sm font-medium">Sex</label>
                <select id="sex" name="sex" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div>
                <label for="cp" class="block text-sm font-medium">Chest Pain Type</label>
                <select id="cp" name="cp" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="1">Typical Angina</option>
                    <option value="2">Atypical Angina</option>
                    <option value="3">Non-anginal Pain</option>
                    <option value="4">Asymptomatic</option>
                </select>
            </div>

            <div>
                <label for="trestbps" class="block text-sm font-medium">Resting BP</label>
                <input type="number" id="trestbps" name="trestbps" required placeholder="e.g., 120"
                    class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md">
            </div>

            <div>
                <label for="chol" class="block text-sm font-medium">Cholesterol</label>
                <input type="number" id="chol" name="chol" required placeholder="e.g., 211"
                    class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md">
            </div>

            <div>
                <label for="fbs" class="block text-sm font-medium">Fasting Blood Sugar</label>
                <select id="fbs" name="fbs" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="1">&gt; 120 mg/dL</option>
                    <option value="0">&lt;= 120 mg/dL</option>
                </select>
            </div>

            <div>
                <label for="restecg" class="block text-sm font-medium">Resting ECG</label>
                <select id="restecg" name="restecg" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="0">Normal</option>
                    <option value="1">ST-T wave abnormality</option>
                    <option value="2">Left ventricular hypertrophy</option>
                </select>
            </div>

            <div>
                <label for="thalach" class="block text-sm font-medium">Max Heart Rate</label>
                <input type="number" id="thalach" name="thalach" required placeholder="e.g., 142"
                    class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md">
            </div>

            <div>
                <label for="exang" class="block text-sm font-medium">Exercise Induced Angina</label>
                <select id="exang" name="exang" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div>
                <label for="oldpeak" class="block text-sm font-medium">Oldpeak</label>
                <input type="number" step="0.1" id="oldpeak" name="oldpeak" required placeholder="e.g., 1.8"
                    class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md">
            </div>

            <div>
                <label for="slope" class="block text-sm font-medium">ST Slope</label>
                <select id="slope" name="slope" required class="mt-1 block w-full border-gray-300 rounded-md">
                    <option value="1">Upsloping</option>
                    <option value="2">Flat</option>
                    <option value="3">Downsloping</option>
                </select>
            </div>

            <div class="md:col-span-2 lg:col-span-3 text-center mt-4">
                <button type="submit" class="w-full sm:w-auto py-3 px-8 text-white bg-indigo-600 rounded-md hover:bg-indigo-700 transition">
                    Predict
                </button>
            </div>
        </form>
    </main>

    <section id="result-section" class="mt-8 text-center hidden">
        <h2 class="text-2xl font-semibold">Prediction Result</h2>
        <div id="result-box" class="mt-4 p-6 rounded-2xl text-white text-xl font-bold">
            <p id="result-text"></p>
            <p id="probability-text" class="text-sm font-normal mt-1"></p>
        </div>
    </section>

</div>

<script>
    const form = document.getElementById('prediction-form');
    const resultSection = document.getElementById('result-section');
    const resultBox = document.getElementById('result-box');
    const resultText = document.getElementById('result-text');
    const probabilityText = document.getElementById('probability-text');

    // quick logistic regression mock model
    const model = {
        weights: {
            age: 0.03,
            sex: 0.3,
            cp: 0.4,
            trestbps: 0.01,
            chol: 0.005,
            fbs: 0.1,
            restecg: 0.2,
            thalach: -0.04,
            exang: 0.5,
            oldpeak: 0.6,
            slope: 0.4
        },
        bias: -10.5
    };

    const sigmoid = z => 1 / (1 + Math.exp(-z));

    form.addEventListener('submit', e => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(form).entries());
        for (let key in data) data[key] = parseFloat(data[key]);

        // weighted sum
        let z = model.bias;
        for (let feature in model.weights) {
            z += model.weights[feature] * data[feature];
        }

        const probability = sigmoid(z);
        const prediction = probability >= 0.5 ? 1 : 0;

        showResult(prediction, probability);
    });

    function showResult(pred, prob) {
        resultSection.classList.remove('hidden');
        const percent = (prob * 100).toFixed(2);
        
        if (pred) {
            resultBox.classList.remove('bg-green-500');
            resultBox.classList.add('bg-red-500');
            resultText.textContent = 'Heart Disease Detected';
        } else {
            resultBox.classList.remove('bg-red-500');
            resultBox.classList.add('bg-green-500');
            resultText.textContent = 'No Heart Disease Detected';
        }
        probabilityText.textContent = `Model Confidence: ${percent}%`;
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
</script>
</body>
</html>
