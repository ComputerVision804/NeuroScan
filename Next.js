"use client";
import { useState } from "react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h2 className="text-xl mb-4">Upload Brain MRI</h2>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
      >
        Predict
      </button>

      {result && (
        <div className="mt-6 p-4 border rounded shadow">
          <h3 className="font-bold">Prediction: {result.prediction}</h3>
          <ul>
            {Object.entries(result.probabilities).map(([label, prob]) => (
              <li key={label}>{label}: {(Number(prob) * 100).toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
