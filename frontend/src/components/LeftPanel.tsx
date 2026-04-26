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
import type { AppState, DrawnStation, Mode, Prediction, TrainService } from '../App'

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

const SERVICE_RULE_COPY: Record<
  TrainService,
  { min: string; max: string; description: string; chip: string; glow: string }
> = {
  local: {
    min: '0.2 mi',
    max: '0.75 mi',
    description: 'Frequent neighborhood stops with tighter spacing.',
    chip: 'bg-cyan-400/18 text-cyan-100 ring-cyan-300/25',
    glow: 'from-cyan-400 via-sky-500 to-blue-600',
  },
  express: {
    min: '0.5 mi',
    max: '3.0 mi',
    description: 'Longer spacing for faster cross-borough travel.',
    chip: 'bg-orange-400/18 text-orange-100 ring-orange-300/25',
    glow: 'from-orange-400 via-orange-500 to-amber-500',
  },
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
      className="group flex items-center gap-3 rounded-2xl border border-white/8 bg-[linear-gradient(180deg,rgba(20,25,37,0.98),rgba(13,16,25,0.96))] px-3 py-3 shadow-[0_12px_24px_rgba(0,0,0,0.18)] transition-colors hover:border-white/14"
    >
      <button
        {...attributes}
        {...listeners}
        className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-white/5 text-slate-500 transition-colors hover:bg-white/8 hover:text-slate-200"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <rect x="2" y="3" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="6.25" width="10" height="1.5" rx="0.75" />
          <rect x="2" y="9.5" width="10" height="1.5" rx="0.75" />
        </svg>
      </button>
      <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-white/10 bg-white/[0.04] text-xs font-semibold text-slate-300">
        {index + 1}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-white">{station.name}</p>
        <p className="mt-0.5 text-[11px] uppercase tracking-[0.18em] text-slate-500">
          {station.isNew ? 'Custom stop' : 'Existing station'}
        </p>
      </div>
      <span
        className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] ring-1 ${
          station.isNew
            ? 'bg-orange-400/18 text-orange-100 ring-orange-300/25'
            : 'bg-emerald-400/18 text-emerald-100 ring-emerald-300/25'
        }`}
      >
        {station.isNew ? 'new' : 'live'}
      </span>
      <button
        onClick={() => onRemove(station.id)}
        className="grid h-8 w-8 shrink-0 place-items-center rounded-full text-slate-500 transition-colors hover:bg-red-500/12 hover:text-red-300"
      >
        ✕
      </button>
    </div>
  )
}

function MetricCard({
  label,
  value,
  detail,
  gradient,
}: {
  label: string
  value: string
  detail: string
  gradient: string
}) {
  return (
    <div className="relative overflow-hidden rounded-[1.35rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,23,34,0.98),rgba(11,14,22,0.98))] p-4 shadow-[0_18px_38px_rgba(0,0,0,0.24)]">
      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${gradient}`} />
      <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">{label}</p>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-white">{value}</p>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{detail}</p>
    </div>
  )
}

function ResultsPanel({ prediction }: { prediction: Prediction }) {
  const maxLineDelta = Math.max(
    ...prediction.affected_lines.map(line => Math.abs(line.delta_pct)),
    1
  )

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Simulation Results</p>
          <h3 className="mt-1 text-lg font-semibold tracking-tight text-white">Impact Snapshot</h3>
        </div>
      </div>

      <div className="grid gap-3">
        <MetricCard
          label="Projected Ridership"
          value={prediction.new_line_ridership.toLocaleString()}
          detail={`Daily riders · peak hour ${prediction.peak_hour_ridership.toLocaleString()}`}
          gradient="from-cyan-400 via-sky-500 to-blue-600"
        />
        <MetricCard
          label="Monthly Cost"
          value={fmtCost(prediction.operational_cost_monthly)}
          detail={`Estimated monthly operating cost. Daily equivalent ${fmtCost(prediction.operational_cost_daily)}.`}
          gradient="from-amber-300 via-orange-400 to-orange-500"
        />
      </div>

      {prediction.route_comparison && (
        <div className="overflow-hidden rounded-[1.45rem] border border-white/8 bg-[linear-gradient(180deg,rgba(19,22,35,0.98),rgba(11,14,23,0.98))] shadow-[0_20px_42px_rgba(0,0,0,0.28)]">
          <div className="border-b border-white/8 px-4 py-3">
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Travel Time Comparison</p>
            <h4 className="mt-1 text-base font-semibold text-white">
              {prediction.route_comparison.origin_name} to {prediction.route_comparison.destination_name}
            </h4>
            {!prediction.route_comparison.available && (
              <p className="mt-1 text-xs text-amber-200">
                Existing-network timing is estimated. Proposed corridor timing is predicted for this route.
              </p>
            )}
          </div>
          <div className="grid gap-3 p-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Existing Fastest</p>
              <div className="mt-3 space-y-2 text-sm leading-relaxed text-slate-200">
                <p>
                  Board the <span className="font-semibold text-cyan-300">{prediction.route_comparison.first_train}</span> train
                </p>
                {prediction.route_comparison.transfer_station && prediction.route_comparison.second_train ? (
                  <>
                    <p>
                      Transfer at <span className="font-semibold text-white">{prediction.route_comparison.transfer_station}</span>
                    </p>
                    <p>
                      Continue on the <span className="font-semibold text-cyan-300">{prediction.route_comparison.second_train}</span> train
                    </p>
                  </>
                ) : (
                  <p className="text-emerald-300">Direct ride with no transfer.</p>
                )}
              </div>
              <p className="mt-4 text-3xl font-semibold tracking-tight text-white">
                {prediction.route_comparison.existing_travel_minutes} min
              </p>
            </div>

            <div className="rounded-2xl border border-cyan-300/14 bg-[linear-gradient(180deg,rgba(18,36,54,0.44),rgba(12,19,32,0.34))] p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Proposed Corridor</p>
              <p className="mt-3 text-sm leading-relaxed text-slate-200">
                Direct service from <span className="font-semibold text-white">{prediction.route_comparison.origin_name}</span> to{' '}
                <span className="font-semibold text-white">{prediction.route_comparison.destination_name}</span>.
              </p>
              <p className="mt-4 text-3xl font-semibold tracking-tight text-white">
                {prediction.route_comparison.new_route_minutes} min
              </p>
              {prediction.route_comparison.time_saved_minutes > 0 ? (
                <div className="mt-3 inline-flex rounded-full border border-emerald-300/18 bg-emerald-400/10 px-3 py-1 text-[11px] font-semibold text-emerald-200">
                  Saves about {prediction.route_comparison.time_saved_minutes} minutes
                </div>
              ) : (
                <div className="mt-3 inline-flex rounded-full border border-slate-300/18 bg-slate-400/10 px-3 py-1 text-[11px] font-semibold text-slate-200">
                  Similar travel time to current network
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {prediction.affected_lines.length > 0 && (
        <div className="rounded-[1.35rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.96),rgba(11,14,22,0.98))] p-4 shadow-[0_18px_38px_rgba(0,0,0,0.24)]">
          <div className="mb-4">
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Network Response</p>
            <h4 className="mt-1 text-base font-semibold text-white">Most Affected Lines</h4>
          </div>
          <div className="space-y-3">
            {prediction.affected_lines.slice(0, 5).map(line => {
              const positive = line.delta_pct >= 0
              const barPct = (Math.abs(line.delta_pct) / maxLineDelta) * 100
              return (
                <div key={line.line} className="rounded-2xl border border-white/6 bg-white/[0.03] px-3 py-3">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm font-medium text-white">{line.line} train</span>
                    <span className={`text-sm font-semibold ${positive ? 'text-emerald-300' : 'text-rose-300'}`}>
                      {positive ? '+' : ''}
                      {line.delta_pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800">
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
  setMode: (mode: Mode) => void
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
  setMode,
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
    mode,
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
  const ruleCopy = SERVICE_RULE_COPY[trainService]

  function handleImportChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) importJsonLine(file)
    event.target.value = ''
  }

  return (
    <aside className="relative flex h-screen w-[430px] shrink-0 flex-col overflow-hidden border-r border-white/8 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.12),transparent_28%),radial-gradient(circle_at_top_right,rgba(249,115,22,0.1),transparent_24%),linear-gradient(180deg,#0f1320_0%,#090c13_100%)]">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent_18%,transparent_82%,rgba(255,255,255,0.02))]" />

      <div className="relative border-b border-white/8 px-6 pb-5 pt-6">
        <div className="flex items-start gap-4">
          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-[linear-gradient(135deg,#ef4444,#dc2626)] shadow-[0_14px_30px_rgba(239,68,68,0.34)]">
            <span className="text-[11px] font-black uppercase tracking-[0.16em] text-white">MTA</span>
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[11px] uppercase tracking-[0.28em] text-slate-500">Transit Sandbox</p>
            <h1 className="mt-1 text-[1.9rem] font-semibold tracking-tight text-white">LineLab</h1>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              Sketch a new subway corridor, test stop spacing, and compare it with the existing NYC network.
            </p>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-3 gap-3">
          <div className="rounded-2xl border border-white/8 bg-white/[0.04] px-3 py-3">
            <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Stops</p>
            <p className="mt-2 text-xl font-semibold text-white">{drawnLine.length}</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-white/[0.04] px-3 py-3">
            <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Length</p>
            <p className="mt-2 text-xl font-semibold text-white">{kmTotal.toFixed(1)} km</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-white/[0.04] px-3 py-3">
            <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Pattern</p>
            <p className="mt-2 text-xl font-semibold capitalize text-white">{trainService}</p>
          </div>
        </div>
      </div>

      <div className="relative px-6 pt-5">
        <div className="grid grid-cols-2 rounded-2xl border border-white/8 bg-black/18 p-1.5">
          <button
            onClick={() => setMode('draw')}
            className={`rounded-[1rem] px-4 py-3 text-sm font-semibold transition-all ${
              mode === 'draw'
                ? 'bg-[linear-gradient(135deg,#38bdf8,#2563eb)] text-white shadow-[0_14px_30px_rgba(37,99,235,0.34)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Build Line
          </button>
          <button
            onClick={() => prediction && setMode('results')}
            disabled={!prediction}
            className={`rounded-[1rem] px-4 py-3 text-sm font-semibold transition-all ${
              mode === 'results'
                ? 'bg-[linear-gradient(135deg,#10b981,#0891b2)] text-white shadow-[0_14px_30px_rgba(16,185,129,0.24)]'
                : prediction
                  ? 'text-slate-400 hover:text-slate-200'
                  : 'cursor-not-allowed text-slate-600'
            }`}
          >
            Analyze Impact
          </button>
        </div>
      </div>

      <div ref={scrollRef} className="relative flex-1 space-y-4 overflow-y-auto px-6 pb-6 pt-5">
        <section className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.96),rgba(11,14,22,0.96))] p-4 shadow-[0_18px_38px_rgba(0,0,0,0.22)]">
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Build Instructions</p>
          <div className="mt-3 space-y-2 text-sm leading-relaxed text-slate-300">
            {drawnLine.length === 0 ? (
              <>
                <p>Click an existing NYC station marker to add it to your corridor.</p>
                <p>Right-click within the NYC map area to place a custom station.</p>
                <p>Use the spacing rings to keep the proposed route feasible.</p>
              </>
            ) : (
              <>
                <p>
                  Current alignment has <span className="font-semibold text-white">{drawnLine.length} stops</span>{' '}
                  across <span className="font-semibold text-white">{kmTotal.toFixed(1)} km</span>.
                </p>
                <p>Drag rows to reorder or remove a stop to test alternate alignments.</p>
                <p className="text-slate-400">
                  {trainService} service requires each segment to stay between {ruleCopy.min} and {ruleCopy.max}.
                </p>
              </>
            )}
          </div>
        </section>

        <section className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.96),rgba(11,14,22,0.98))] p-4 shadow-[0_18px_38px_rgba(0,0,0,0.22)]">
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Service Pattern</p>
          <h3 className="mt-1 text-lg font-semibold text-white">Operating style</h3>
          <div className="mt-4">
            <div className="relative grid grid-cols-2 rounded-2xl border border-white/8 bg-white/[0.04] p-1.5">
              <div
                className={`absolute inset-y-1.5 w-[calc(50%-6px)] rounded-[0.95rem] transition-transform duration-300 ease-out ${
                  trainService === 'local'
                    ? 'translate-x-0 bg-[linear-gradient(135deg,rgba(34,211,238,0.34),rgba(37,99,235,0.26))] shadow-[0_12px_26px_rgba(37,99,235,0.18)]'
                    : 'translate-x-[calc(100%+0px)] bg-[linear-gradient(135deg,rgba(251,146,60,0.3),rgba(234,88,12,0.24))] shadow-[0_12px_26px_rgba(249,115,22,0.18)]'
                }`}
              />
              <button
                onClick={() => setTrainService('local')}
                className={`relative z-10 rounded-[0.95rem] px-4 py-3 text-sm font-semibold uppercase tracking-[0.18em] transition-colors ${
                  trainService === 'local' ? 'text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Local
              </button>
              <button
                onClick={() => setTrainService('express')}
                className={`relative z-10 rounded-[0.95rem] px-4 py-3 text-sm font-semibold uppercase tracking-[0.18em] transition-colors ${
                  trainService === 'express' ? 'text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Express
              </button>
            </div>
          </div>
          <div className={`mt-4 inline-flex rounded-full px-3 py-1 text-[11px] font-semibold ring-1 ${ruleCopy.chip}`}>
            Valid spacing: {ruleCopy.min} to {ruleCopy.max}
          </div>
          <p className="mt-3 text-sm leading-relaxed text-slate-400">{ruleCopy.description}</p>
        </section>

        {validationError && (
          <section className="rounded-[1.4rem] border border-amber-300/18 bg-[linear-gradient(180deg,rgba(120,53,15,0.24),rgba(69,26,3,0.28))] p-4 shadow-[0_18px_36px_rgba(120,53,15,0.16)]">
            <p className="text-[11px] uppercase tracking-[0.22em] text-amber-300">Spacing Conflict</p>
            <p className="mt-2 text-sm leading-relaxed text-amber-100">{validationError}</p>
          </section>
        )}

        {drawnLine.length > 0 && (
          <section className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(180deg,rgba(18,22,34,0.98),rgba(10,13,20,0.98))] p-4 shadow-[0_20px_42px_rgba(0,0,0,0.24)]">
            <div className="mb-4 flex items-end justify-between">
              <div>
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Line Assembly</p>
                <h3 className="mt-1 text-lg font-semibold text-white">Stop Sequence</h3>
              </div>
              <p className="text-xs text-slate-500">Drag to reorder</p>
            </div>

            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <SortableContext items={drawnLine.map(station => station.id)} strategy={verticalListSortingStrategy}>
                <div className="space-y-2.5">
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

        {mode === 'results' && prediction && (
          <section className="rounded-[1.55rem] border border-white/8 bg-[linear-gradient(180deg,rgba(14,18,27,0.98),rgba(8,11,18,0.98))] p-4 shadow-[0_22px_48px_rgba(0,0,0,0.28)]">
            <ResultsPanel prediction={prediction} />
          </section>
        )}
      </div>

      <div className="relative border-t border-white/8 bg-[linear-gradient(180deg,rgba(11,13,21,0.84),rgba(8,10,16,0.98))] px-6 pb-6 pt-5">
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={undoLast}
            disabled={drawnLine.length === 0}
            className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-medium text-slate-200 transition-colors hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-35"
          >
            Undo Last
          </button>
          <button
            onClick={clearAll}
            disabled={drawnLine.length === 0 && !prediction}
            className="rounded-2xl border border-red-300/18 bg-red-500/[0.04] px-4 py-3 text-sm font-medium text-rose-300 transition-colors hover:bg-red-500/[0.1] disabled:cursor-not-allowed disabled:opacity-35"
          >
            Clear Line
          </button>
        </div>
        <button
          onClick={predict}
          disabled={!canPredict}
          className={`mt-3 flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-4 text-base font-semibold transition-all ${
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
              Running Forecast…
            </>
          ) : validationError ? (
            'Resolve Stop Spacing'
          ) : (
            'Predict Impact'
          )}
        </button>
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
          className={`mt-3 flex w-full items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-semibold transition-all ${
            loading
              ? 'cursor-not-allowed border-slate-700/80 bg-slate-900/70 text-slate-500'
              : 'border-cyan-300/24 bg-cyan-400/[0.08] text-cyan-100 hover:bg-cyan-400/[0.14]'
          }`}
        >
          Import JSON Line
        </button>
        {prediction && (
          <button
            onClick={exportJsonLine}
            disabled={loading}
            className={`mt-3 flex w-full items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-semibold transition-all ${
              loading
                ? 'cursor-not-allowed border-slate-700/80 bg-slate-900/70 text-slate-500'
                : 'border-violet-300/24 bg-violet-400/[0.08] text-violet-100 hover:bg-violet-400/[0.14]'
            }`}
          >
            Export JSON
          </button>
        )}
        <button
          onClick={suggestedLine ? toggleSuggestedLineView : suggestCheaperLine}
          disabled={!canPredict}
          className={`mt-3 flex w-full items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-semibold transition-all ${
            canPredict
              ? 'border-emerald-300/28 bg-emerald-500/[0.08] text-emerald-200 hover:bg-emerald-500/[0.14]'
              : 'cursor-not-allowed border-slate-700/80 bg-slate-900/70 text-slate-500'
          }`}
        >
          {suggestedLine
            ? showingSuggestedLine
              ? 'Show Your Line'
              : 'Show Suggested Line'
            : 'Suggest Cheaper Line'}
        </button>
        {suggestionSummary && (
          <div className="mt-3 rounded-2xl border border-emerald-300/22 bg-emerald-500/[0.08] px-3 py-3">
            <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-300">Why Suggested</p>
            <p className="mt-1 text-xs leading-relaxed text-emerald-100">{suggestionSummary}</p>
          </div>
        )}
      </div>
    </aside>
  )
}
