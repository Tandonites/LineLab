import { useRef } from 'react'
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import type { AppState, DrawnStation, Mode, Prediction } from '../App'

// ── Helpers ───────────────────────────────────────────────────────────────────
function haversineKm(a: DrawnStation, b: DrawnStation): number {
  const R = 6371
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLon = ((b.lon - a.lon) * Math.PI) / 180
  const lat1 = (a.lat * Math.PI) / 180
  const lat2 = (b.lat * Math.PI) / 180
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x))
}

function totalKm(line: DrawnStation[]): number {
  let km = 0
  for (let i = 1; i < line.length; i++) km += haversineKm(line[i - 1], line[i])
  return km
}

function fmtCost(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}

// ── Sortable station row ──────────────────────────────────────────────────────
function SortableStationRow({
  station,
  index,
  onRemove,
}: {
  station: DrawnStation
  index: number
  onRemove: (id: string) => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: station.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-[#1a1d27] border border-gray-700/50 group"
    >
      {/* Drag handle */}
      <button
        {...attributes}
        {...listeners}
        className="text-gray-600 hover:text-gray-400 cursor-grab active:cursor-grabbing touch-none"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <rect x="2" y="3" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="6.25" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="9.5" width="10" height="1.5" rx="0.75" />
        </svg>
      </button>
      <span className="text-gray-500 text-xs w-4 shrink-0">{index + 1}</span>
      <span className="text-white text-xs flex-1 truncate">{station.name}</span>
      <span
        className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${
          station.isNew
            ? 'bg-orange-900/60 text-orange-300'
            : 'bg-emerald-900/60 text-emerald-300'
        }`}
      >
        {station.isNew ? 'new' : 'existing'}
      </span>
      <button
        onClick={() => onRemove(station.id)}
        className="text-gray-600 hover:text-red-400 ml-1 transition-colors"
      >
        ✕
      </button>
    </div>
  )
}

// ── Results section ───────────────────────────────────────────────────────────
function ResultsPanel({ prediction }: { prediction: Prediction }) {
  const maxLineDelta = Math.max(
    ...prediction.affected_lines.map(l => Math.abs(l.delta_pct)),
    1
  )

  return (
    <div className="space-y-3">
      <div className="border-t border-gray-700 pt-4">
        <p className="text-gray-400 text-xs uppercase tracking-widest mb-3 font-medium">
          Simulation Results
        </p>
      </div>

      {/* New line ridership */}
      <div className="bg-[#1a1d27] border-l-4 border-blue-500 rounded-r-xl p-3">
        <p className="text-gray-400 text-xs mb-1">New Line Ridership</p>
        <p className="text-white text-2xl font-bold">
          {prediction.new_line_ridership.toLocaleString()}
        </p>
        <p className="text-gray-500 text-xs mt-0.5">
          daily riders · peak hour:{' '}
          <span className="text-gray-300">
            {prediction.peak_hour_ridership.toLocaleString()}
          </span>
        </p>
      </div>

      {/* Operational cost */}
      <div className="bg-[#1a1d27] border-l-4 border-yellow-500 rounded-r-xl p-3">
        <p className="text-gray-400 text-xs mb-1">Operational Cost</p>
        <p className="text-white text-2xl font-bold">
          {fmtCost(prediction.operational_cost_daily)}
        </p>
        <p className="text-gray-500 text-xs mt-0.5">estimated daily operating cost</p>
      </div>

      {/* Affected lines */}
      {prediction.affected_lines.length > 0 && (
        <div className="bg-[#1a1d27] border-l-4 border-red-500 rounded-r-xl p-3">
          <p className="text-gray-400 text-xs mb-2">Most Affected Lines</p>
          <div className="space-y-2">
            {prediction.affected_lines.slice(0, 5).map(l => {
              const positive = l.delta_pct >= 0
              const barPct = (Math.abs(l.delta_pct) / maxLineDelta) * 100
              return (
                <div key={l.line}>
                  <div className="flex justify-between items-center mb-0.5">
                    <span className="text-white text-xs font-semibold">
                      {l.line} train
                    </span>
                    <span
                      className={`text-xs font-medium ${
                        positive ? 'text-emerald-400' : 'text-red-400'
                      }`}
                    >
                      {positive ? '+' : ''}
                      {l.delta_pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1 w-full bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        positive ? 'bg-emerald-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${barPct}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  state: AppState
  setMode: (mode: Mode) => void
  removeStation: (id: string) => void
  undoLast: () => void
  clearAll: () => void
  reorderLine: (newOrder: DrawnStation[]) => void
  predict: () => void
}

// ── Main component ────────────────────────────────────────────────────────────
export default function LeftPanel({
  state,
  setMode,
  removeStation,
  undoLast,
  clearAll,
  reorderLine,
  predict,
}: Props) {
  const { mode, drawnLine, loading, prediction } = state
  const scrollRef = useRef<HTMLDivElement>(null)

  const sensors = useSensors(useSensor(PointerSensor))

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (over && active.id !== over.id) {
      const oldIdx = drawnLine.findIndex(s => s.id === active.id)
      const newIdx = drawnLine.findIndex(s => s.id === over.id)
      reorderLine(arrayMove(drawnLine, oldIdx, newIdx))
    }
  }

  const canPredict = drawnLine.length >= 2 && !loading
  const kmTotal = totalKm(drawnLine)

  return (
    <div className="w-[380px] shrink-0 h-screen bg-[#0f1117] border-r border-gray-800 flex flex-col">
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="px-5 pt-5 pb-4 border-b border-gray-800 flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-full bg-[#EE352E] flex items-center justify-center shrink-0">
          <span className="text-white text-[9px] font-black tracking-tight">MTA</span>
        </div>
        <div>
          <h1 className="text-white text-[18px] font-semibold leading-tight">
            Transit Predictor
          </h1>
          <p className="text-gray-400 text-[12px]">MTA New Line Simulator</p>
        </div>
      </div>

      {/* ── Mode toggle ─────────────────────────────────────────────────────── */}
      <div className="px-4 pt-4 shrink-0">
        <div className="flex gap-2 bg-[#1a1d27] p-1 rounded-xl">
          <button
            onClick={() => setMode('draw')}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              mode === 'draw'
                ? 'bg-blue-600 text-white shadow'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Draw Line
          </button>
          <button
            onClick={() => prediction && setMode('results')}
            disabled={!prediction}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              mode === 'results'
                ? 'bg-blue-600 text-white shadow'
                : prediction
                ? 'text-gray-400 hover:text-gray-200'
                : 'text-gray-600 cursor-not-allowed'
            }`}
          >
            View Results
          </button>
        </div>
      </div>

      {/* ── Scrollable body ─────────────────────────────────────────────────── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {/* Instructions card */}
        <div className="bg-[#1a1d27] border border-gray-700/50 rounded-xl p-3 text-xs text-gray-300 leading-relaxed">
          {drawnLine.length === 0 ? (
            <>
              Left click an existing station marker to add it to your line.
              Right click anywhere on the map to place a new station.
              Press <kbd className="bg-gray-700 px-1 rounded">Undo</kbd> to remove the last point.
            </>
          ) : (
            <>
              <span className="text-white font-medium">{drawnLine.length} station{drawnLine.length !== 1 ? 's' : ''}</span>
              {drawnLine.length >= 2 && (
                <> · <span className="text-blue-400">{kmTotal.toFixed(1)} km</span> total route length</>
              )}
              <br />
              Drag to reorder. Click ✕ to remove a station.
            </>
          )}
        </div>

        {/* Station list */}
        {drawnLine.length > 0 && (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={drawnLine.map(s => s.id)}
              strategy={verticalListSortingStrategy}
            >
              <div className="space-y-1.5">
                {drawnLine.map((station, i) => (
                  <SortableStationRow
                    key={station.id}
                    station={station}
                    index={i}
                    onRemove={removeStation}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        )}

        {/* Results */}
        {mode === 'results' && prediction && (
          <div
            className="overflow-hidden transition-all duration-300"
            style={{ maxHeight: prediction ? 9999 : 0 }}
          >
            <ResultsPanel prediction={prediction} />
          </div>
        )}
      </div>

      {/* ── Action buttons ───────────────────────────────────────────────────── */}
      <div className="px-4 pb-5 pt-3 border-t border-gray-800 space-y-2 shrink-0">
        <div className="flex gap-2">
          <button
            onClick={undoLast}
            disabled={drawnLine.length === 0}
            className="flex-1 py-2 rounded-xl text-xs font-medium bg-[#1a1d27] text-gray-300 hover:bg-[#2a2d3a] disabled:opacity-40 disabled:cursor-not-allowed border border-gray-700 transition-colors"
          >
            Undo Last
          </button>
          <button
            onClick={clearAll}
            disabled={drawnLine.length === 0 && !prediction}
            className="flex-1 py-2 rounded-xl text-xs font-medium border border-red-800 text-red-400 hover:bg-red-900/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Clear All
          </button>
        </div>
        <button
          onClick={predict}
          disabled={!canPredict}
          className={`w-full py-2.5 rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
            canPredict
              ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-900/40'
              : 'bg-blue-900/30 text-blue-400/40 cursor-not-allowed'
          }`}
        >
          {loading ? (
            <>
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle
                  className="opacity-25"
                  cx="12" cy="12" r="10"
                  stroke="currentColor" strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
              Predicting…
            </>
          ) : (
            'Predict Impact'
          )}
        </button>
      </div>
    </div>
  )
}
