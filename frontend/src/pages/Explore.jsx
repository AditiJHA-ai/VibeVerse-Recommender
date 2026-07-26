import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Nav from '../components/Nav'

const API = '/api'

function ResultCard({ item, i }) {
  return (
    <article className="rec-item" style={{ animationDelay: `${i * 0.05}s` }}>
      <h3>{item.title}</h3>
      <p className="creator">by {item.creator}</p>
      <div className="match-bar">
        <span style={{ width: `${Math.round(Math.min(1, item.similarity) * 100)}%` }} />
      </div>
      <div className="match-label">
        Match: {Math.round(item.similarity * 100)}%
        {item.primary_vibe ? ` · ${item.primary_vibe}` : ''}
      </div>
      {item.why && <p className="why">{item.why}</p>}
    </article>
  )
}

function shorten(text, max = 720) {
  if (!text) return ''
  const clean = text.replace(/\s+/g, ' ').trim()
  if (clean.length <= max) return clean
  const chunk = clean.slice(0, max)
  const ends = [chunk.lastIndexOf('. '), chunk.lastIndexOf('! '), chunk.lastIndexOf('? ')]
  const cut = Math.max(...ends)
  if (cut > 120) return `${chunk.slice(0, cut + 1).trim()}`
  // avoid mid-word cuts
  const space = chunk.lastIndexOf(' ')
  return `${(space > 120 ? chunk.slice(0, space) : chunk).trim()}…`
}

export default function Explore() {
  const [searchParams] = useSearchParams()
  const bootstrapped = useRef(false)
  const [have, setHave] = useState('book')
  const [want, setWant] = useState('all')
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (query.trim().length < 2) {
      setSuggestions([])
      return
    }
    const t = setTimeout(async () => {
      try {
        const res = await fetch(
          `${API}/suggest?q=${encodeURIComponent(query.trim())}&type=${have}`,
        )
        const data = await res.json()
        setSuggestions(data.results || [])
      } catch {
        setSuggestions([])
      }
    }, 220)
    return () => clearTimeout(t)
  }, [query, have])

  useEffect(() => {
    if (bootstrapped.current) return
    const q = (searchParams.get('q') || '').trim()
    const haveParam = searchParams.get('have')
    if (!q) return
    bootstrapped.current = true
    const nextHave = haveParam === 'song' || haveParam === 'book' ? haveParam : 'book'
    setHave(nextHave)
    setQuery(q)
    // Defer so state above is visible; call recommend with explicit args.
    ;(async () => {
      setLoading(true)
      setError('')
      setResult(null)
      try {
        const res = await fetch(`${API}/recommend`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            input_type: nextHave,
            want: 'all',
            top_n: 8,
            allow_live: true,
          }),
        })
        const data = await res.json().catch(() => ({}))
        if (!res.ok) {
          throw new Error(
            typeof data.detail === 'string'
              ? data.detail
              : "We couldn't find that title. Try a fuller name.",
          )
        }
        setResult(data)
        setSuggestions([])
      } catch (err) {
        setError(err.message || 'Something went wrong')
      } finally {
        setLoading(false)
      }
    })()
  }, [searchParams])

  const books = useMemo(
    () => (result?.recommendations || []).filter((r) => r.type === 'book'),
    [result],
  )
  const songs = useMemo(
    () => (result?.recommendations || []).filter((r) => r.type === 'song'),
    [result],
  )

  async function runRecommend(q = query) {
    const trimmed = q.trim()
    if (!trimmed) {
      setError(`Type a ${have} title first.`)
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await fetch(`${API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: trimmed,
          input_type: have,
          want,
          top_n: 8,
          allow_live: true,
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : "We couldn't find that title. Try a fuller name.",
        )
      }
      setResult(data)
      setSuggestions([])
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    runRecommend()
  }

  const matchedLabel = result?.matched_type === 'book' ? 'book' : 'song'
  const blurb = shorten(result?.description || '')

  return (
    <div className="explore-page">
      <Nav />
      <div className="site-shell explore">
      <header className="explore-hero">
        <h1>Find your vibe</h1>
        <p>
          Pick whether you&apos;re starting from a <strong>book</strong> or a{' '}
          <strong>song</strong>, then we&apos;ll find matches with a similar emotional mood.
        </p>
      </header>

      <form className="search-panel" id="search" onSubmit={onSubmit}>
        <div className="have-toggle" role="tablist" aria-label="What are you starting from?">
          <button
            type="button"
            className={have === 'book' ? 'active' : ''}
            onClick={() => {
              setHave('book')
              setQuery('')
              setResult(null)
              setSuggestions([])
            }}
          >
            📚 I have a book
          </button>
          <button
            type="button"
            className={have === 'song' ? 'active' : ''}
            onClick={() => {
              setHave('song')
              setQuery('')
              setResult(null)
              setSuggestions([])
            }}
          >
            🎵 I have a song
          </button>
        </div>

        <label className="field-label" htmlFor="title-input">
          {have === 'book' ? 'Choose a book…' : 'Choose a song…'}
        </label>
        <div className="search-row single">
          <input
            id="title-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              have === 'book'
                ? 'e.g. Fourth Wing, Dune Messiah, The Odyssey…'
                : 'e.g. song title or “song name - artist”…'
            }
            aria-label={have === 'book' ? 'Book title' : 'Song title'}
          />
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Finding matches…' : 'Find my vibe'}
          </button>
        </div>

        {suggestions.length > 0 && (
          <div className="suggestions">
            <span className="suggest-hint">Suggestions:</span>
            {suggestions.map((s) => (
              <button
                type="button"
                className="chip"
                key={`${s.type}-${s.title}`}
                onClick={() => {
                  setQuery(s.title)
                  runRecommend(s.title)
                }}
              >
                {s.title}
                {s.creator ? ` - ${s.creator}` : ''}
              </button>
            ))}
          </div>
        )}

        <div className="want-fieldset" role="group" aria-labelledby="want-label">
          <p className="want-label" id="want-label">
            What do you want recommendations for?
          </p>
          <label className="radio">
            <input
              type="radio"
              name="want"
              checked={want === 'all'}
              onChange={() => setWant('all')}
            />
            All
          </label>
          <label className="radio">
            <input
              type="radio"
              name="want"
              checked={want === 'book'}
              onChange={() => setWant('book')}
            />
            Books
          </label>
          <label className="radio">
            <input
              type="radio"
              name="want"
              checked={want === 'song'}
              onChange={() => setWant('song')}
            />
            Songs
          </label>
        </div>
      </form>

      {error && <p className="error">{error}</p>}
      {loading && <p className="status">Matching emotional vibe…</p>}

      {result && (
        <>
          <div className="meta-banner">
            <p className="eyebrow">Because you like this {matchedLabel}</p>
            <div className="meta-row">
              {result.thumbnail && (
                <img className="cover" src={result.thumbnail} alt="" />
              )}
              <div>
                <h2>{result.matched_title}</h2>
                <p className="byline">
                  by {result.matched_creator || 'Unknown'}
                  {result.primary_vibe
                    ? ` · mood: ${result.primary_vibe}${
                        result.vibe_labels ? ` (${result.vibe_labels})` : ''
                      }`
                    : ''}
                </p>
                {blurb ? (
                  <p className="explain">{blurb}</p>
                ) : (
                  <p className="explain">
                    A {result.primary_vibe || 'rich'} {matchedLabel}
                    {result.vibe_labels ? ` with notes of ${result.vibe_labels}` : ''}.
                  </p>
                )}
              </div>
            </div>
          </div>

          {(want === 'all' || want === 'book') && (
            <section className="rec-section">
              <h3 className="rec-heading">📚 Book matches</h3>
              {books.length === 0 ? (
                <p className="status">No strong book matches for this vibe.</p>
              ) : (
                <div className="rec-grid">
                  {books.map((item, i) => (
                    <ResultCard item={item} i={i} key={`b-${item.title}-${i}`} />
                  ))}
                </div>
              )}
            </section>
          )}

          {(want === 'all' || want === 'song') && (
            <section className="rec-section">
              <h3 className="rec-heading">🎧 Song matches</h3>
              {songs.length === 0 ? (
                <p className="status">No strong song matches for this vibe.</p>
              ) : (
                <div className="rec-grid">
                  {songs.map((item, i) => (
                    <ResultCard item={item} i={i} key={`s-${item.title}-${i}`} />
                  ))}
                </div>
              )}
            </section>
          )}
        </>
      )}
      </div>
    </div>
  )
}
