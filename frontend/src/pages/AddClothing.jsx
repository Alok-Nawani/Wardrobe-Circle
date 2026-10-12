/**
 * src/pages/AddClothing.jsx
 * ==========================
 * Editorial Add Clothing page acting as an AI Scanner.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadZone from '../components/UploadZone.jsx';
import AnalysisCard from '../components/AnalysisCard.jsx';
import { Toast, useToast } from '../components/ui/Toast.jsx';
import { addWardrobeItem } from '../api/wardrobeApi.js';
import { analyzeClothing, findSimilarClothing } from '../api/clothingApi.js';
import { useWardrobeContext } from '../context/WardrobeContext.jsx';

export default function AddClothing() {
  const navigate = useNavigate();
  const { addItem } = useWardrobeContext();
  const { toast, show, hide } = useToast();

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState(null);
  const [savedItem, setSavedItem] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [similar, setSimilar] = useState(null);

  const handleFile = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setSavedItem(null);
    setAnalysis(null);
    setSimilar(null);
    setError(null);
  };

  const handleAnalyzeAndSave = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      setStatus('Analyzing signature...');
      const analysisResult = await analyzeClothing(file);
      setAnalysis(analysisResult);

      setStatus('Cross-referencing catalog...');
      findSimilarClothing(file, 5).then(setSimilar).catch(() => {});

      setStatus('Adding to collection...');
      const record = await addWardrobeItem(file);
      setSavedItem(record);
      addItem(record);
      show('Piece added to your collection');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setStatus('');
    }
  };

  return (
    <div className="flex flex-col gap-10 animate-fade-in max-w-5xl mx-auto">
      {toast && <Toast message={toast.message} type={toast.type} onClose={hide} />}

      <section className="text-center max-w-2xl mx-auto space-y-4">
        <p className="editorial-subheading">Scanner</p>
        <h1 className="editorial-heading">Add to your collection</h1>
        <p className="text-brand-subtext">Upload a piece you own. The AI extracts its visual signature automatically.</p>
      </section>

      {!file && (
        <section className="mt-8 animate-slide-up">
          <UploadZone onFile={handleFile} />
        </section>
      )}

      {file && (
        <section className="grid lg:grid-cols-2 gap-12 items-start mt-8">
          {/* LEFT: Image */}
          <div className="clothing-image-container aspect-[3/4] sticky top-32 animate-zoom-in">
            <img src={preview} alt="Upload" className="clothing-image" />
          </div>

          {/* RIGHT: Analysis / Actions */}
          <div className="flex flex-col gap-10 animate-slide-up">
            {!savedItem && !analysis && (
              <div className="flex flex-col items-start gap-6 border-l border-brand-border pl-8 py-4">
                <h3 className="font-serif text-2xl">Ready for analysis</h3>
                <p className="text-sm text-brand-subtext">{file.name} • {(file.size/1024).toFixed(0)} KB</p>
                <div className="flex gap-4">
                  <button onClick={handleAnalyzeAndSave} disabled={loading} className="btn-primary">
                    {loading ? status : 'Analyze & Save'}
                  </button>
                  <button onClick={() => setFile(null)} disabled={loading} className="btn-ghost">Discard</button>
                </div>
              </div>
            )}

            {error && <div className="text-red-600 text-sm border-l-2 border-red-600 pl-4">{error}</div>}

            {analysis && (
              <AnalysisCard analysis={analysis} similar={similar} saved={!!savedItem} />
            )}

            {savedItem && (
              <div className="flex gap-4 pt-8 border-t border-brand-border mt-8">
                <button onClick={() => navigate('/wardrobe')} className="btn-primary">View in Wardrobe</button>
                <button onClick={() => setFile(null)} className="btn-secondary">Add Another</button>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
