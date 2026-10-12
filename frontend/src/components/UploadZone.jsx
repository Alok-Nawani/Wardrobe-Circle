/**
 * src/components/UploadZone.jsx
 * ==============================
 * Premium drag-and-drop upload zone.
 */

import { useCallback, useState } from 'react';

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];

export default function UploadZone({ onFile, disabled = false }) {
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const validate = (file) => {
    if (!ACCEPTED.includes(file.type)) {
      setValidationError('Please upload a JPEG or PNG image.');
      return false;
    }
    if (file.size > 20 * 1024 * 1024) {
      setValidationError('File is too large. Maximum size is 20 MB.');
      return false;
    }
    setValidationError(null);
    return true;
  };

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (validate(file)) onFile(file);
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div className="w-full">
      <label
        className={`relative flex flex-col items-center justify-center
          min-h-[400px] border border-brand-border bg-[#EAE7DF] cursor-pointer
          transition-all duration-500 ease-editorial
          ${dragging ? 'bg-[#DFDBD1] border-brand-accent scale-[1.01]' : 'hover:bg-[#E2DFD6]'}
          ${disabled ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        aria-label="Upload clothing image"
      >
        <div className="flex flex-col items-center gap-6 text-center z-10 px-6">
          <span className="font-serif text-3xl md:text-4xl text-brand-text">
            Drop your piece here.
          </span>
          <div className="flex flex-col items-center gap-2">
            <span className="text-sm uppercase tracking-widest font-semibold border-b border-brand-text pb-1">
              Or browse files
            </span>
            <span className="text-[10px] uppercase tracking-widest text-brand-subtext mt-4">
              JPEG · PNG · UP TO 20 MB
            </span>
          </div>
        </div>

        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={onInputChange}
          className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-20"
          aria-label="File input"
          disabled={disabled}
        />
      </label>

      {validationError && (
        <p className="mt-4 text-xs text-red-600 text-center uppercase tracking-widest">
          {validationError}
        </p>
      )}
    </div>
  );
}
