import { Link } from 'react-router-dom'
import Nav from '../components/Nav'
import heroImg from '../assets/hero.png'

const VIBES = [
  { name: 'Intimate', desc: 'Cozy, introspective, deeply personal connections' },
  { name: 'Electric', desc: 'Energetic, thrilling, adrenaline-pumping intensity' },
  { name: 'Dreamy', desc: 'Ethereal, imaginative, wonderfully surreal' },
  { name: 'Epic', desc: 'Grand, ambitious, sweeping and transformative' },
]

const PAIRINGS = [
  {
    title: 'The Great Gatsby',
    body: 'Book vibe: jazz-age glamour with melancholy undertones. Music match: atmospheric, nostalgic pop that captures the same bittersweet longing.',
  },
  {
    title: 'Midnights by Taylor Swift',
    body: 'Song vibe: introspective late-night reflections. Book match: the poetic intimacy and emotional depth of Sally Rooney’s Normal People.',
  },
  {
    title: 'Dune by Frank Herbert',
    body: 'Book vibe: epic world-building with orchestral grandeur. Music match: sweeping atmospheric scores that echo the same sense of wonder and scale.',
  },
]

const REASONS = [
  {
    title: 'Escape algorithm predictability',
    body: 'Traditional recommendations keep you in comfortable corners. VibeVerse introduces fresh perspectives from completely different mediums.',
  },
  {
    title: 'Deepen your emotional connection',
    body: 'When a song captures exactly what a book made you feel, the experience becomes richer — kindred spirits across art forms.',
  },
  {
    title: 'Expand your creative horizons',
    body: 'Art forms inform each other. Discovering how your favorite music connects to literature opens new ways of thinking about both.',
  },
]

export default function Landing() {
  return (
    <div className="site-shell landing">
      <div className="hero-plane" aria-hidden="true">
        <img src={heroImg} alt="" />
        <div className="hero-veil" />
      </div>

      <Nav />

      <header className="hero">
        <p className="hero-brand">
          Vibe<span>Verse</span>
        </p>
        <h2>Discover the soundtrack to your favorite book.</h2>
        <p>
          Find your next literary obsession through music. VibeVerse bridges the gap between
          melodies and pages, revealing the hidden connections between stories and songs.
        </p>
        <div className="cta-row">
          <Link className="btn btn-primary" to="/explore">
            Explore Vibes
          </Link>
          <a className="btn btn-ghost" href="#how">
            How it works
          </a>
        </div>
      </header>

      <section className="section" id="how">
        <h3>What&apos;s your vibe?</h3>
        <p className="lead">
          What if your favorite book had a soundtrack? What if that song you love was actually
          a novel? By analyzing emotional essence, mood, and atmosphere, VibeVerse uncovers
          connections traditional recommendations miss.
        </p>

        <h3 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.2rem)', marginTop: '2rem' }}>
          Two questions, infinite possibilities
        </h3>
        <div className="two-col" style={{ marginTop: '1.25rem', marginBottom: '2.5rem' }}>
          <div className="mode-card">
            <div className="emoji">📚</div>
            <h4>Book to Song</h4>
            <p>
              Take a novel you love and discover its perfect musical match. What melody captures
              the essence of your favorite story?
            </p>
          </div>
          <div className="mode-card">
            <div className="emoji">🎵</div>
            <h4>Song to Book</h4>
            <p>
              Start with a song that moves you and find the literary companion that complements
              its vibe. Your next page-turner awaits.
            </p>
          </div>
        </div>

        <h3 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.2rem)' }}>How the magic happens</h3>
        <div className="steps" style={{ marginTop: '1.25rem' }}>
          <div className="step">
            <div className="num">01</div>
            <h4>Analyze the vibe</h4>
            <p>Mood, tone, pacing, themes, and emotional arcs — the true character of each work.</p>
          </div>
          <div className="step">
            <div className="num">02</div>
            <h4>Cross-domain matching</h4>
            <p>Books and songs share one vibe vocabulary, so emotional signals actually transfer.</p>
          </div>
          <div className="step">
            <div className="num">03</div>
            <h4>Surface perfect pairs</h4>
            <p>Unexpected connections that feel surprisingly right.</p>
          </div>
        </div>
      </section>

      <section className="section">
        <h3>Vibe categories</h3>
        <p className="lead">
          Every book and song lives somewhere on the VibeVerse spectrum — from cozy and
          introspective to epic and exhilarating.
        </p>
        <div className="vibe-grid">
          {VIBES.map((v) => (
            <div className="vibe-pill" key={v.name}>
              <strong>{v.name}</strong>
              <span>{v.desc}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h3>Try these pairings</h3>
        <p className="lead">Real examples of how books and songs connect across the vibe spectrum.</p>
        <div className="pairings">
          {PAIRINGS.map((p) => (
            <article className="pairing" key={p.title}>
              <h4>{p.title}</h4>
              <p>{p.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <h3>Why cross-media matters</h3>
        <p className="lead">Stop siloing your entertainment — discovery knows no boundaries.</p>
        <div className="reasons">
          {REASONS.map((r) => (
            <article className="reason" key={r.title}>
              <h4>{r.title}</h4>
              <p>{r.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <h3>Ready to explore?</h3>
        <p className="lead">
          Enter a favorite book or song and let VibeVerse reveal connections you never imagined.
        </p>
        <div className="cta-row">
          <Link className="btn btn-primary" to="/explore">
            Start with what you love
          </Link>
        </div>
      </section>

      <footer className="footer">
        <div className="brand">
          Vibe<span>Verse</span>
        </div>
        Where every song has a story. Where every book has a soundtrack. Where discovery knows no
        boundaries.
      </footer>
    </div>
  )
}
