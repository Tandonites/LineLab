import { useEffect, useRef } from 'react'

interface Props {
  message: string
  onDismiss: () => void
}

export default function Toast({ message, onDismiss }: Props) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    timerRef.current = setTimeout(onDismiss, 4000)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [onDismiss])

  return (
    <div className="fixed bottom-6 right-6 z-[900] flex items-center gap-3 bg-red-950 border border-red-700 text-red-200 text-sm px-4 py-3 rounded-xl shadow-2xl animate-fade-in max-w-sm">
      <svg className="w-4 h-4 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      </svg>
      <span className="flex-1">{message}</span>
      <button onClick={onDismiss} className="text-red-400 hover:text-red-200 transition-colors ml-1">✕</button>
    </div>
  )
}
