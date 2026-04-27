import { useRef, type ChangeEvent } from 'react'
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
import type { AppState, DrawnStation, Prediction, TrainService } from '../App'

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
  for (let i = 1; i < line.length; i += 1) km += haversineKm(line[i - 1], line[i])
  return km
}

function fmtCost(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}

function TrainBadge({ line }: { line: string }) {
  return (
    <span className="inline-flex items-center rounded-md bg-cyan-400/15 px-1.5 py-0.5 text-[11px] font-bold text-cyan-200 ring-1 ring-cyan-300/25">
      {line}
    </span>
  )
}

function RouteDisplay({
  firstTrain,
  transferStation,
  secondTrain,
  isWalkingOnly,
}: {
  firstTrain: string
  transferStation: string | null
  secondTrain: string | null
  isWalkingOnly: boolean
}) {
  if (isWalkingOnly) {
    return <span className="text-xs text-amber-300">Walking only</span>
  }
  if (transferStation && secondTrain) {
    return (
      <span className="flex flex-wrap items-center gap-1 text-xs text-slate-400">
        <TrainBadge line={firstTrain} />
        <span>→ {transferStation} →</span>
        <TrainBadge line={secondTrain} />
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-xs text-slate-400">
      <TrainBadge line={firstTrain} />
      <span className="text-slate-500">Direct</span>
    </span>
  )
}

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
    opacity: isDragging ? 0.55 : 1,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="group flex items-center gap-3 rounded-2xl border border-white/8 bg-[linear-gradient(180deg,rgba(20,25,37,0.98),rgba(13,16,25,0.96))] px-3 py-2.5 shadow-[0_12px_24px_rgba(0,0,0,0.18)] transition-colors hover:border-white/14"
    >
      <button
        {...attributes}
        {...listeners}
        className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-slate-600 transition-colors hover:text-slate-300"
      >
        <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
          <rect x="2" y="3" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="6.25" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="9.5" width="10" height="1.5" rx="0.75" />
        </svg>
      </button>
      <div className="grid h-7 w-7 shrink-0 place-items-center rounded-full border border-white/10 bg-white/[0.04] text-[11px] font-semibold text-slate-400">
        {index + 1}
      </div>
      <p className="min-w-0 flex-1 truncate text-sm font-medium text-white">{station.name}</p>
      <span
        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] ring-1 ${
          station.isNew
            ? 'bg-orange-400/18 text-orange-100 ring-orange-300/25'
            : 'bg-emerald-400/18 text-emerald-100 ring-emerald-300/25'
        }`}
      >
        {station.isNew ? 'new' : 'live'}
      </span>
      <button
        onClick={() => onRemove(station.id)}
        className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-slate-600 transition-colors hover:bg-red-500/12 hover:text-red-300"
      >
        ✕
      </button>
    </div>
  )
}

function ResultsPanel({ prediction }: { prediction: Prediction }) {
  const maxLineDelta = Math.max(
    ...prediction.affected_lines.map(line => Math.abs(line.delta_pct)),
    1
  )

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div className="relative overflow-hidden rounded-[1.2rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,23,34,0.98),rgba(11,14,22,0.98))] p-4">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-cyan-400 via-sky-500 to-blue-600" />
          <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Daily Riders</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-white">
            {prediction.new_line_ridership.toLocaleString()}
          </p>
          <p className="mt-1 text-[11px] text-slate-500">
            peak {prediction.peak_hour_ridership.toLocaleString()}/hr
          </p>
        </div>
        <div className="relative overflow-hidden rounded-[1.2rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,23,34,0.98),rgba(11,14,22,0.98))] p-4">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-amber-300 via-orange-400 to-orange-500" />
          <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Monthly Cost</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-white">
            {fmtCost(prediction.operational_cost_monthly)}
          </p>
          <p className="mt-1 text-[11px] text-slate-500">
            daily {fmtCost(prediction.operational_cost_daily)}
          </p>
        </div>
      </div>

      {prediction.route_comparison && (
        <div className="overflow-hidden rounded-[1.35rem] border border-white/8 bg-[linear-gradient(180deg,rgba(19,22,35,0.98),rgba(11,14,23,0.98))]">
          <div className="border-b border-white/8 px-4 py-3">
            <p className="text-xs font-medium text-slate-300">
              {prediction.route_comparison.origin_name}
              <span className="mx-1.5 text-slate-600">→</span>
              {prediction.route_comparison.destination_name}
            </p>
          </div>
          <div className="grid grid-cols-2 divide-x divide-white/8">
            <div className="p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Existing</p>
              <p className="mt-3 text-3xl font-semibold tracking-tight text-white">
                {prediction.route_comparison.existing_travel_minutes}
                <span className="ml-1 text-base font-normal text-slate-400">min</span>
              </p>
              <div className="mt-2">
                <RouteDisplay
                  firstTrain={prediction.route_comparison.first_train}
                  transferStation={prediction.route_comparison.transfer_station ?? null}
                  secondTrain={prediction.route_comparison.second_train ?? null}
                  isWalkingOnly={prediction.route_comparison.is_walking_only}
                />
              </div>
            </div>
            <div className="p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Proposed</p>
              <p className="mt-3 text-3xl font-semibold tracking-tight text-white">
                {prediction.route_comparison.new_route_minutes}
                <span className="ml-1 text-base font-normal text-slate-400">min</span>
              </p>
              <div className="mt-2">
                {prediction.route_comparison.time_saved_minutes > 0 ? (
                  <span className="inline-flex rounded-full bg-emerald-400/12 px-2 py-0.5 text-[11px] font-semibold text-emerald-300 ring-1 ring-emerald-300/20">
                    −{prediction.route_comparison.time_saved_minutes} min
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-500">≈ same time</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {prediction.affected_lines.length > 0 && (
        <div className="rounded-[1.2rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.96),rgba(11,14,22,0.98))] p-4">
          <p className="mb-3 text-[10px] uppercase tracking-[0.2em] text-slate-500">Network Impact</p>
          <div className="space-y-2.5">
            {prediction.affected_lines.slice(0, 5).map(line => {
              const positive = line.delta_pct >= 0
              const barPct = (Math.abs(line.delta_pct) / maxLineDelta) * 100
              return (
                <div key={line.line} className="flex items-center gap-3">
                  <span className="w-8 shrink-0 text-sm font-medium text-white">{line.line}</span>
                  <div className="flex-1">
                    <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full ${
                          positive
                            ? 'bg-[linear-gradient(90deg,#10b981,#34d399)]'
                            : 'bg-[linear-gradient(90deg,#ef4444,#fb7185)]'
                        }`}
                        style={{ width: `${barPct}%` }}
                      />
                    </div>
                  </div>
                  <span className={`w-14 text-right text-xs font-semibold ${positive ? 'text-emerald-300' : 'text-rose-300'}`}>
                    {positive ? '+' : ''}{line.delta_pct.toFixed(1)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

interface Props {
  state: AppState
  setTrainService: (trainService: TrainService) => void
  removeStation: (id: string) => void
  undoLast: () => void
  clearAll: () => void
  reorderLine: (newOrder: DrawnStation[]) => void
  predict: () => void
  importJsonLine: (file: File) => void
  exportJsonLine: () => void
  suggestCheaperLine: () => void
  toggleSuggestedLineView: () => void
}

export default function LeftPanel({
  state,
  setTrainService,
  removeStation,
  undoLast,
  clearAll,
  reorderLine,
  predict,
  importJsonLine,
  exportJsonLine,
  suggestCheaperLine,
  toggleSuggestedLineView,
}: Props) {
  const {
    drawnLine,
    loading,
    prediction,
    suggestionSummary,
    validationError,
    trainService,
    suggestedLine,
    showingSuggestedLine,
  } = state
  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const sensors = useSensors(useSensor(PointerSensor))

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (over && active.id !== over.id) {
      const oldIdx = drawnLine.findIndex(station => station.id === active.id)
      const newIdx = drawnLine.findIndex(station => station.id === over.id)
      reorderLine(arrayMove(drawnLine, oldIdx, newIdx))
    }
  }

  const canPredict = drawnLine.length >= 2 && !loading && !validationError
  const kmTotal = totalKm(drawnLine)

  function handleImportChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) importJsonLine(file)
    event.target.value = ''
  }

  return (
    <aside className="relative flex h-screen w-[400px] shrink-0 flex-col overflow-hidden border-r border-white/8 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.12),transparent_28%),radial-gradient(circle_at_top_right,rgba(249,115,22,0.1),transparent_24%),linear-gradient(180deg,#0f1320_0%,#090c13_100%)]">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent_18%,transparent_82%,rgba(255,255,255,0.02))]" />

      {/* Header */}
      <div className="relative border-b border-white/8 px-5 pb-4 pt-5">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[linear-gradient(135deg,#ef4444,#dc2626)] shadow-[0_10px_24px_rgba(239,68,68,0.30)]">
            <span className="text-[10px] font-black uppercase tracking-[0.14em] text-white">MTA</span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">LineLab</h1>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <div className="rounded-xl border border-white/8 bg-white/[0.04] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Stops</p>
            <p className="mt-1 text-lg font-semibold text-white">{drawnLine.length}</p>
          </div>
          <div className="rounded-xl border border-white/8 bg-white/[0.04] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Length</p>
            <p className="mt-1 text-lg font-semibold text-white">{kmTotal.toFixed(1)} km</p>
          </div>
          <div className="rounded-xl border border-white/8 bg-white/[0.04] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Pattern</p>
            <p className="mt-1 text-lg font-semibold capitalize text-white">{trainService}</p>
          </div>
        </div>
      </div>


      <div ref={scrollRef} className="relative flex-1 space-y-3 overflow-y-auto px-5 pb-5 pt-4">
        {/* Service pattern */}
        <div className="flex items-center gap-3 rounded-2xl border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.96),rgba(11,14,22,0.98))] px-4 py-3">
          <div className="relative grid flex-1 grid-cols-2 rounded-xl border border-white/8 bg-white/[0.04] p-1">
            <div
              className={`absolute inset-y-1 w-[calc(50%-4px)] rounded-[0.6rem] transition-transform duration-300 ease-out ${
                trainService === 'local'
                  ? 'translate-x-0 bg-[linear-gradient(135deg,rgba(34,211,238,0.34),rgba(37,99,235,0.26))]'
                  : 'translate-x-[calc(100%+0px)] bg-[linear-gradient(135deg,rgba(251,146,60,0.3),rgba(234,88,12,0.24))]'
              }`}
            />
            <button
              onClick={() => setTrainService('local')}
              className={`relative z-10 rounded-[0.6rem] py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition-colors ${
                trainService === 'local' ? 'text-white' : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              Local
            </button>
            <button
              onClick={() => setTrainService('express')}
              className={`relative z-10 rounded-[0.6rem] py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition-colors ${
                trainService === 'express' ? 'text-white' : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              Express
            </button>
          </div>
          <span className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${
            trainService === 'local'
              ? 'bg-cyan-400/18 text-cyan-100 ring-cyan-300/25'
              : 'bg-orange-400/18 text-orange-100 ring-orange-300/25'
          }`}>
            {trainService === 'local' ? '0.2–0.75 mi' : '0.5–3.0 mi'}
          </span>
        </div>

        {validationError && (
          <div className="rounded-2xl border border-amber-300/18 bg-[linear-gradient(180deg,rgba(120,53,15,0.24),rgba(69,26,3,0.28))] px-4 py-3">
            <p className="text-[10px] uppercase tracking-[0.2em] text-amber-300">Spacing Conflict</p>
            <p className="mt-1 text-xs leading-relaxed text-amber-100">{validationError}</p>
          </div>
        )}

        {drawnLine.length === 0 && (
          <div className="rounded-2xl border border-white/6 bg-white/[0.02] px-4 py-5 text-center">
            <p className="text-sm text-slate-400">
              <span className="font-medium text-white">Left-click</span> a station to add it to your line.
            </p>
            <p className="mt-2 text-sm text-slate-400">
              <span className="font-medium text-white">Right-click</span> anywhere to place a custom stop.
            </p>
          </div>
        )}

        {drawnLine.length > 0 && (
          <section className="rounded-2xl border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.98),rgba(10,13,20,0.98))] p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Stop Sequence</p>
              <p className="text-[10px] text-slate-600">Drag to reorder</p>
            </div>

            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <SortableContext items={drawnLine.map(station => station.id)} strategy={verticalListSortingStrategy}>
                <div className="space-y-2">
                  {drawnLine.map((station, index) => (
                    <SortableStationRow
                      key={station.id}
                      station={station}
                      index={index}
                      onRemove={removeStation}
                    />
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          </section>
        )}

        {prediction && (
          <section className="rounded-2xl border border-white/8 bg-[linear-gradient(180deg,rgba(14,18,27,0.98),rgba(8,11,18,0.98))] p-4">
            <ResultsPanel prediction={prediction} />
          </section>
        )}
      </div>

      {/* Bottom actions */}
      <div className="relative border-t border-white/8 bg-[linear-gradient(180deg,rgba(11,13,21,0.84),rgba(8,10,16,0.98))] px-5 pb-5 pt-4">
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={undoLast}
            disabled={drawnLine.length === 0}
            className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-slate-300 transition-colors hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-35"
          >
            Undo
          </button>
          <button
            onClick={clearAll}
            disabled={drawnLine.length === 0 && !prediction}
            className="rounded-xl border border-red-300/18 bg-red-500/[0.04] px-4 py-2.5 text-sm font-medium text-rose-300 transition-colors hover:bg-red-500/[0.1] disabled:cursor-not-allowed disabled:opacity-35"
          >
            Clear
          </button>
        </div>

        <button
          onClick={predict}
          disabled={!canPredict}
          className={`mt-2 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3.5 text-sm font-semibold transition-all ${
            canPredict
              ? trainService === 'local'
                ? 'bg-[linear-gradient(135deg,#22d3ee,#2563eb)] text-white shadow-[0_18px_40px_rgba(37,99,235,0.26)] hover:scale-[1.01]'
                : 'bg-[linear-gradient(135deg,#fb923c,#ea580c)] text-white shadow-[0_18px_40px_rgba(249,115,22,0.24)] hover:scale-[1.01]'
              : 'cursor-not-allowed bg-slate-800/80 text-slate-500'
          }`}
        >
          {loading ? (
            <>
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Running…
            </>
          ) : validationError ? (
            'Fix Stop Spacing'
          ) : (
            'Predict Impact'
          )}
        </button>

        <div className="mt-2 grid grid-cols-2 gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={handleImportChange}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="rounded-xl border border-cyan-300/24 bg-cyan-400/[0.07] px-3 py-2.5 text-xs font-semibold text-cyan-200 transition-colors hover:bg-cyan-400/[0.13] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Import JSON
          </button>
          <button
            onClick={exportJsonLine}
            disabled={loading || !prediction}
            className="rounded-xl border border-violet-300/24 bg-violet-400/[0.07] px-3 py-2.5 text-xs font-semibold text-violet-200 transition-colors hover:bg-violet-400/[0.13] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Export JSON
          </button>
        </div>

        <button
          onClick={suggestedLine ? toggleSuggestedLineView : suggestCheaperLine}
          disabled={!canPredict}
          className={`mt-2 flex w-full items-center justify-center rounded-xl border px-4 py-2.5 text-xs font-semibold transition-all ${
            canPredict
              ? 'border-emerald-300/28 bg-emerald-500/[0.07] text-emerald-200 hover:bg-emerald-500/[0.13]'
              : 'cursor-not-allowed border-slate-700/80 bg-slate-900/70 text-slate-500'
          }`}
        >
          {suggestedLine
            ? showingSuggestedLine
              ? 'Show My Line'
              : 'Show Suggested Line'
            : 'Suggest Cheaper Line'}
        </button>

        {suggestionSummary && (
          <p className="mt-2 text-[11px] leading-relaxed text-emerald-300/80 px-1">{suggestionSummary}</p>
        )}
      </div>
    </aside>
  )
}
