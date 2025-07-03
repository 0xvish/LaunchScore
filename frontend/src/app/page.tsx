"use client";
import { useState, ChangeEvent, FormEvent, FocusEvent } from "react";
// import Image from "next/image";

interface PredictResult {
  ml_score: number;
  llm_score: number;
  final_score: number;
  llm_analysis: string;
  error?: string;
}

interface FieldErrors {
  [key: string]: boolean;
}

// Simple markdown parser for basic formatting
const parseMarkdown = (text: string): string => {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/^- (.*$)/gim, "<li>$1</li>")
    .replace(/^(\d+\. .*$)/gim, "<li>$1</li>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(.*)$/, "<p>$1</p>")
    .replace(/<p><\/p>/g, "")
    .replace(/(<li>[\s\S]*<\/li>)/g, "<ul>$1</ul>");
};

// Predefined options based on ML model training data
const SECTORS = [
  "FinTech",
  "HealthTech",
  "EdTech",
  "E-commerce",
  "SaaS",
  "AI/ML",
  "IoT",
  "Blockchain",
  "AgriTech",
  "FoodTech",
  "PropTech",
  "RetailTech",
  "Logistics",
  "Gaming",
  "Media",
  "CleanTech",
  "SpaceTech",
  "Biotech",
  "Marketplace",
  "B2B",
  "Consumer",
  "Enterprise",
  "Mobile",
  "Web3",
];

const FUNDING_STAGES = [
  "Not Funded",
  "Bootstrapped",
  "Pre-Seed",
  "Seed",
  "Series A",
  "Series B",
  "Series C",
  "Series D+",
  "Angel",
  "Bridge",
  "Convertible",
  "Debt",
  "Grant",
  "Revenue Based",
];

const HEADQUARTERS = [
  "Bengaluru",
  "Mumbai",
  "Delhi",
  "Hyderabad",
  "Chennai",
  "Pune",
  "Gurugram",
  "Noida",
  "Kolkata",
  "Ahmedabad",
  "Jaipur",
  "Kochi",
  "Indore",
  "Lucknow",
  "Bhubaneswar",
  "Coimbatore",
  "Vadodara",
  "Nagpur",
  "Thiruvananthapuram",
  "Chandigarh",
  "Mysuru",
  "Visakhapatnam",
  "Surat",
  "Kanpur",
];

export default function Home() {
  const [form, setForm] = useState({
    idea: "",
    sector: "",
    stage: "",
    headquarter: "",
    founded: 2020,
    amount: 10000,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});

  const validateField = (
    field: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
  ): boolean => {
    const { name, value, type } = field;

    if (field.hasAttribute("required") && !value.trim()) {
      setFieldErrors((prev) => ({ ...prev, [name]: true }));
      return false;
    } else if (
      type === "number" &&
      value &&
      field instanceof HTMLInputElement
    ) {
      const numValue = parseFloat(value);
      const minValue = field.min ? parseFloat(field.min) : null;
      const maxValue = field.max ? parseFloat(field.max) : null;

      if (minValue && numValue < minValue) {
        setFieldErrors((prev) => ({ ...prev, [name]: true }));
        return false;
      } else if (maxValue && numValue > maxValue) {
        setFieldErrors((prev) => ({ ...prev, [name]: true }));
        return false;
      }
    }

    setFieldErrors((prev) => ({ ...prev, [name]: false }));
    return true;
  };

  const validateForm = (): boolean => {
    let isValid = true;
    const formElement = document.getElementById(
      "predict-form"
    ) as HTMLFormElement;
    if (!formElement) return false;

    const requiredFields = formElement.querySelectorAll(
      "[required]"
    ) as NodeListOf<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>;

    requiredFields.forEach((field) => {
      if (!validateField(field)) {
        isValid = false;
      }
    });

    if (!isValid) {
      // Form validation failed - could add toast notification here if needed
    }

    return isValid;
  };

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    const parsed =
      name === "founded" || name === "amount" ? Number(value) : value;
    setForm((prev) => ({ ...prev, [name]: parsed }));

    // Validate field on change
    validateField(e.target);
  };

  const handleBlur = (
    e:
      | ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
      | FocusEvent<HTMLSelectElement>
  ) => {
    validateField(
      e.target as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    );
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    // Validate form before submission
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setResult(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data: PredictResult = await res.json();
      setResult(data);
    } catch (err) {
      console.error("API request failed:", err);
      setResult({
        error: "Request failed",
        ml_score: 0,
        llm_score: 0,
        final_score: 0,
        llm_analysis: "",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-gray-50 via-white to-slate-50 text-gray-900 font-sans min-h-screen flex flex-col items-center px-3 sm:px-4 py-6 sm:py-10 relative overflow-x-hidden">
      {/* Minimal background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-4 -left-4 w-24 h-24 sm:w-32 sm:h-32 bg-slate-100 rounded-full opacity-10 animate-bounce"></div>
        <div className="absolute top-1/4 -right-8 w-16 h-16 sm:w-20 sm:h-20 bg-gray-100 rounded-full opacity-15 animate-pulse"></div>
        <div
          className="absolute bottom-1/4 -left-6 w-20 h-20 sm:w-28 sm:h-28 bg-slate-100 rounded-full opacity-10 animate-bounce"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="absolute -bottom-4 -right-4 w-18 h-18 sm:w-24 sm:h-24 bg-gray-100 rounded-full opacity-12 animate-pulse"
          style={{ animationDelay: "2s" }}
        ></div>
      </div>

      <div className="w-full max-w-xs sm:max-w-md md:max-w-lg lg:max-w-2xl bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 relative z-10">
        {/* Header with minimal styling */}
        <div className="text-center space-y-2">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900">
            🚀 LaunchScore
          </h1>
          <p className="text-xs sm:text-sm text-gray-600">
            Predict your startup&apos;s success with AI-powered analysis
          </p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs sm:text-sm text-gray-700">
            <div className="font-semibold text-blue-800 mb-2">
              🔍 How Our AI Works:
            </div>
            <div className="space-y-1">
              <div>
                <span className="font-medium">💡 Startup Idea:</span> Analyzed
                using{" "}
                <span className="text-blue-700">LLM & Semantic Search</span>
              </div>
              <div>
                <span className="font-medium">📊 Other Fields:</span> Processed
                by{" "}
                <span className="text-blue-700">
                  Neural Network Regression Model
                </span>
              </div>
            </div>
          </div>
        </div>

        <form
          id="predict-form"
          onSubmit={handleSubmit}
          className="space-y-3 sm:space-y-4"
        >
          <div className="group">
            <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
              💡 Startup Idea
              <span className="text-red-500">*</span>
              <span className="text-blue-600 text-xs font-normal">
                (LLM + Semantic Analysis)
              </span>
            </label>
            <textarea
              name="idea"
              rows={3}
              required
              value={form.idea}
              onChange={handleChange}
              onBlur={handleBlur}
              placeholder="Describe your innovative startup idea in detail - this will be analyzed by our AI language model..."
              className={`w-full border ${
                fieldErrors.idea
                  ? "border-red-300 focus:ring-red-300"
                  : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
              } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400 resize-none`}
            />
            {fieldErrors.idea && (
              <div className="text-red-500 text-xs mt-1">
                Please describe your startup idea
              </div>
            )}
            <div className="text-xs text-blue-600 mt-1">
              📝 This field uses advanced NLP and semantic search to understand
              your idea&apos;s potential
            </div>
          </div>

          {/* Two column layout for larger screens */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-3">
            <div className="text-xs text-green-700">
              🤖 <span className="font-semibold">Neural Network Fields:</span>{" "}
              The following fields are processed by our regression model trained
              on startup success patterns
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <div className="group">
              <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
                🏢 Sector
                <span className="text-red-500">*</span>
                <span className="text-green-600 text-xs font-normal">
                  (NN Model)
                </span>
              </label>
              <select
                name="sector"
                required
                value={form.sector}
                onChange={handleChange}
                onBlur={handleBlur}
                className={`w-full border ${
                  fieldErrors.sector
                    ? "border-red-300 focus:ring-red-300"
                    : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
                } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400 bg-white`}
              >
                <option value="">Select a sector...</option>
                {SECTORS.map((sector) => (
                  <option key={sector} value={sector}>
                    {sector}
                  </option>
                ))}
              </select>
              {fieldErrors.sector && (
                <div className="text-red-500 text-xs mt-1">
                  Please select a sector
                </div>
              )}
              <div className="text-xs text-green-600 mt-1">
                💡 Choose from ML-trained sectors for better accuracy
              </div>
            </div>

            <div className="group">
              <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
                📈 Funding Stage
                <span className="text-red-500">*</span>
                <span className="text-green-600 text-xs font-normal">
                  (NN Model)
                </span>
              </label>
              <select
                name="stage"
                required
                value={form.stage}
                onChange={handleChange}
                onBlur={handleBlur}
                className={`w-full border ${
                  fieldErrors.stage
                    ? "border-red-300 focus:ring-red-300"
                    : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
                } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400 bg-white`}
              >
                <option value="">Select funding stage...</option>
                {FUNDING_STAGES.map((stage) => (
                  <option key={stage} value={stage}>
                    {stage}
                  </option>
                ))}
              </select>
              {fieldErrors.stage && (
                <div className="text-red-500 text-xs mt-1">
                  Please select a funding stage
                </div>
              )}
              <div className="text-xs text-green-600 mt-1">
                💡 Choose from ML-trained stages for better accuracy
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <div className="group">
              <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
                🏙️ Headquarter
                <span className="text-red-500">*</span>
                <span className="text-green-600 text-xs font-normal">
                  (NN Model)
                </span>
              </label>
              <select
                name="headquarter"
                required
                value={form.headquarter}
                onChange={handleChange}
                onBlur={handleBlur}
                className={`w-full border ${
                  fieldErrors.headquarter
                    ? "border-red-300 focus:ring-red-300"
                    : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
                } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400 bg-white`}
              >
                <option value="">Select headquarters...</option>
                {HEADQUARTERS.map((hq) => (
                  <option key={hq} value={hq}>
                    {hq}
                  </option>
                ))}
              </select>
              {fieldErrors.headquarter && (
                <div className="text-red-500 text-xs mt-1">
                  Please select a headquarter location
                </div>
              )}
              <div className="text-xs text-green-600 mt-1">
                🇮🇳 Focus on Indian cities for better predictions
              </div>
            </div>

            <div className="group">
              <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
                📅 Founded Year
                <span className="text-red-500">*</span>
                <span className="text-green-600 text-xs font-normal">
                  (NN Model)
                </span>
              </label>
              <input
                type="number"
                name="founded"
                min="2000"
                max="2025"
                required
                value={form.founded}
                onChange={handleChange}
                onBlur={handleBlur}
                placeholder="e.g. 2021"
                className={`w-full border ${
                  fieldErrors.founded
                    ? "border-red-300 focus:ring-red-300"
                    : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
                } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400`}
              />
              {fieldErrors.founded && (
                <div className="text-red-500 text-xs mt-1">
                  Please enter the founded year (2000-2025)
                </div>
              )}
            </div>
          </div>

          <div className="group">
            <label className="block text-xs sm:text-sm font-semibold text-gray-700 mb-1 group-focus-within:text-gray-900 transition-colors duration-200">
              💰 Funding Amount
              <span className="text-red-500">*</span>
              <span className="text-green-600 text-xs font-normal">
                (NN Model)
              </span>
            </label>
            <input
              type="number"
              name="amount"
              min="10000"
              step="10000"
              required
              value={form.amount}
              onChange={handleChange}
              onBlur={handleBlur}
              placeholder="e.g. ₹5000000"
              className={`w-full border ${
                fieldErrors.amount
                  ? "border-red-300 focus:ring-red-300"
                  : "border-gray-300 focus:ring-gray-400 focus:border-gray-400"
              } rounded-lg p-3 text-sm sm:text-base mt-1 focus:ring-2 transition-all duration-300 hover:border-gray-400`}
            />
            {fieldErrors.amount && (
              <div className="text-red-500 text-xs mt-1">
                Please enter the funding amount (minimum ₹10,000)
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gray-900 hover:bg-gray-800 text-white font-semibold px-6 py-3 sm:py-4 rounded-lg transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg active:scale-95 text-sm sm:text-base disabled:opacity-50"
          >
            <span className="flex items-center justify-center space-x-2">
              {loading ? (
                <>
                  <div className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <span>🔮 Predict My Success</span>
              )}
            </span>
          </button>
        </form>

        {(result || loading) && (
          <div className="mt-4 sm:mt-6">
            <div className="bg-gray-50 rounded-xl p-4 sm:p-6 border border-gray-200">
              <h2 className="text-lg sm:text-xl font-bold text-gray-900 mb-3 sm:mb-4 flex items-center">
                <span className="text-xl sm:text-2xl mr-2">📊</span>
                Prediction Results
              </h2>

              {/* Loading state */}
              {loading && (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 sm:h-12 sm:w-12 border-b-2 border-gray-800"></div>
                  <span className="ml-3 text-sm sm:text-base text-gray-600">
                    Analyzing your startup...
                  </span>
                </div>
              )}

              {/* Results content */}
              {result && !loading && (
                <>
                  <div className="prose prose-sm sm:prose-base prose-gray max-w-none bg-white border border-gray-200 rounded-lg p-3 sm:p-4 text-sm sm:text-base">
                    {result.error ? (
                      <div className="text-center py-6">
                        <div className="text-4xl mb-2">❌</div>
                        <div className="text-red-600 font-semibold">
                          Error: Could not get prediction
                        </div>
                        <div className="text-sm text-gray-500 mt-1">
                          Please try again later
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="mb-4">
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
                            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
                              <div className="text-xs text-blue-600 font-semibold">
                                ML Score
                              </div>
                              <div className="text-lg font-bold text-blue-800">
                                {result.ml_score.toFixed(2)}
                              </div>
                            </div>
                            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
                              <div className="text-xs text-green-600 font-semibold">
                                LLM Score
                              </div>
                              <div className="text-lg font-bold text-green-800">
                                {result.llm_score.toFixed(2)}
                              </div>
                            </div>
                            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-center">
                              <div className="text-xs text-purple-600 font-semibold">
                                Final Score
                              </div>
                              <div className="text-lg font-bold text-purple-800">
                                {result.final_score.toFixed(2)}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div
                          className="prose prose-sm max-w-none"
                          dangerouslySetInnerHTML={{
                            __html: parseMarkdown(
                              result.llm_analysis || "No analysis provided."
                            ),
                          }}
                        />
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-6 sm:mt-8 text-center">
        <p className="text-xs sm:text-sm text-gray-500 mb-2">
          Built with ❤️ by{" "}
          <a
            href="https://vishvam.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-700 hover:text-gray-900 transition-colors duration-200 font-medium"
          >
            0xvish
          </a>
        </p>
      </div>
    </div>
  );
}
