import { Link } from 'react-router-dom'
import './Landing.css'
import heroBooks from '../assets/gamma/hero-books.jpg'
import twoQuestions from '../assets/gamma/two-questions.jpg'
import vibeWheel from '../assets/gamma/vibe-wheel.jpg'
import crossDomain from '../assets/gamma/cross-domain.jpg'
import readyExplore from '../assets/gamma/ready-explore.jpg'
import pairingsBg from '../assets/gamma/pairings-bg.jpg'

const PAIRINGS = [
  {
    title: 'The Great Gatsby',
    body: (
      <>
        <strong>Book Vibe:</strong> Jazz-age glamour with melancholy undertones.{' '}
        <strong>Music Match:</strong> Lana Del Rey&apos;s atmospheric, nostalgic pop sensibility
        captures the same bittersweet longing.
      </>
    ),
  },
  {
    title: 'Midnights by Taylor Swift',
    body: (
      <>
        <strong>Song Vibe:</strong> Introspective late-night reflections layered with pop
        production. <strong>Book Match:</strong> The poetic intimacy and emotional depth of Sally
        Rooney&apos;s <em>Normal People</em>.
      </>
    ),
  },
  {
    title: 'Dune by Frank Herbert',
    body: (
      <>
        <strong>Book Vibe:</strong> Epic, expansive world-building with orchestral grandeur.{' '}
        <strong>Music Match:</strong> Hans Zimmer&apos;s sweeping, atmospheric film scores echo
        the same sense of wonder and scale.
      </>
    ),
  },
]

export default function Landing() {
  return (
    <div className="gamma">
      <section className="g-section g-hero">
        <div className="g-wrap g-split">
          <div className="g-copy">
            <h1 className="g-brand">VibeVerse</h1>
            <p className="g-lede">
              Discover the soundtrack to your favorite book. Find your next literary obsession
              through music. VibeVerse bridges the gap between melodies and pages, revealing the
              hidden connections between stories and songs.
            </p>
            <Link className="g-btn" to="/explore">
              Explore Vibes
            </Link>
          </div>
          <figure className="g-art">
            <img src={heroBooks} alt="Floating books and musical notes" />
          </figure>
        </div>
      </section>

      <section className="g-section g-section-tight">
        <div className="g-wrap g-text-block">
          <h2>What&apos;s Your Vibe?</h2>
          <p>
            VibeVerse reimagines how we discover content by asking a delightfully simple question:{' '}
            <strong>what if your favorite book had a soundtrack?</strong> What if that song you
            love was actually a novel? By analyzing the emotional essence, mood, and atmosphere of
            books and songs, VibeVerse uncovers surprising connections across media that traditional
            recommendations miss.
          </p>
        </div>
      </section>

      <section className="g-section">
        <div className="g-wrap g-split g-split-start">
          <div className="g-copy">
            <h2>Two Questions, Infinite Possibilities</h2>
            <div className="g-mode-stack">
              <div className="g-mode-row">
                <span className="g-mode-icon" aria-hidden="true">
                  📚
                </span>
                <div>
                  <h3>Book to Song</h3>
                  <p>
                    Take a novel you love and discover its perfect musical match. What melody
                    captures the essence of your favorite story?
                  </p>
                </div>
              </div>
              <div className="g-mode-row">
                <span className="g-mode-icon" aria-hidden="true">
                  🎵
                </span>
                <div>
                  <h3>Song to Book</h3>
                  <p>
                    Start with a song that moves you and find the literary companion that
                    complements its vibe perfectly. Your next page-turner awaits.
                  </p>
                </div>
              </div>
            </div>
          </div>
          <figure className="g-art">
            <img src={twoQuestions} alt="Cosmic book with swirling musical notes" />
          </figure>
        </div>
      </section>

      <section className="g-section g-band-blue" id="how">
        <div className="g-wrap">
          <h2>How the Magic Happens</h2>
          <p className="g-intro">
            VibeVerse uses sophisticated content analysis to understand the true character of every
            book and song—not just genre tags, but the emotional landscape that makes each work
            unique.
          </p>
          <div className="g-steps">
            <article>
              <div className="g-step-num">01</div>
              <h3>Analyze the Vibe</h3>
              <p>
                We examine mood, tone, pacing, themes, and emotional arcs to capture the true
                essence of each work.
              </p>
            </article>
            <article>
              <div className="g-step-num">02</div>
              <h3>Cross-Domain Matching</h3>
              <p>
                Our algorithm finds items across music and literature that share similar emotional
                resonance and atmosphere.
              </p>
            </article>
            <article>
              <div className="g-step-num">03</div>
              <h3>Surface Perfect Pairs</h3>
              <p>
                Discover unexpected connections that feel surprisingly right—recommendations that
                make you say, &ldquo;I never thought of it that way!&rdquo;
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="g-section">
        <div className="g-wrap">
          <h2>Vibe Categories</h2>
          <p className="g-intro">
            Every book and song lives somewhere on the VibeVerse spectrum. From cozy and
            introspective to epic and exhilarating, we understand the full emotional range of
            storytelling and music.
          </p>
          <div className="g-vibe-ring">
            <div className="g-vibe-label g-vibe-tl">
              <h3>Intimate</h3>
              <p>Cozy, introspective, deeply personal connections</p>
            </div>
            <div className="g-vibe-label g-vibe-tr">
              <h3>Electric</h3>
              <p>Energetic, thrilling, adrenaline-pumping intensity</p>
            </div>
            <img className="g-wheel" src={vibeWheel} alt="Vibe category wheel" />
            <div className="g-vibe-label g-vibe-bl">
              <h3>Epic</h3>
              <p>Grand, ambitious, sweeping and transformative</p>
            </div>
            <div className="g-vibe-label g-vibe-br">
              <h3>Dreamy</h3>
              <p>Ethereal, imaginative, wonderfully surreal</p>
            </div>
          </div>
        </div>
      </section>

      <section className="g-section">
        <div className="g-wrap g-split">
          <div className="g-copy g-copy-short">
            <h2>Cross-Domain Discovery</h2>
            <p>
              Stop siloing your entertainment. Whether you&apos;re a bookworm curious about new
              music or a music lover ready to dive into literature, VibeVerse opens doors to
              discovery you didn&apos;t know existed.
            </p>
          </div>
          <figure className="g-art g-art-wide">
            <img src={crossDomain} alt="Futuristic library and music stage" />
          </figure>
        </div>
      </section>

      <section className="g-section g-pairings">
        <div className="g-pairings-bg" style={{ backgroundImage: `url(${pairingsBg})` }} />
        <div className="g-wrap">
          <div className="g-pairings-panel">
            <h2>Try These Pairings</h2>
            <p className="g-pairings-lead">
              Curious what VibeVerse might reveal? Here are real examples of how books and songs
              connect across the vibe spectrum:
            </p>
            <div className="g-pairings-grid">
              {PAIRINGS.map((item) => (
                <article key={item.title}>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="g-section">
        <div className="g-wrap">
          <h2>Why Cross-Media Matters</h2>
          <div className="g-reasons">
            <article>
              <span className="g-dot" aria-hidden="true" />
              <h3>Escape Algorithm Predictability</h3>
              <p>
                Traditional recommendations keep you in comfortable corners. VibeVerse breaks the
                mold by introducing fresh perspectives from completely different creative mediums.
              </p>
            </article>
            <article>
              <span className="g-dot" aria-hidden="true" />
              <h3>Deepen Your Emotional Connection</h3>
              <p>
                When a song captures exactly what a book made you feel, the experience becomes
                richer. You&apos;re not just finding recommendations—you&apos;re uncovering kindred
                spirits.
              </p>
            </article>
            <article>
              <span className="g-dot" aria-hidden="true" />
              <h3>Expand Your Creative Horizons</h3>
              <p>
                Art forms inform each other. Discovering how your favorite music connects to
                literature opens new ways of thinking about both.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="g-section g-band-sky">
        <div className="g-wrap g-split">
          <div className="g-copy">
            <h2>Ready to Explore?</h2>
            <h3 className="g-subhead">Start with What You Love</h3>
            <p>
              Enter a favorite book or song and let VibeVerse reveal connections you never imagined.
              Every pairing tells a story about emotional resonance and creative kinship.
            </p>
            <ul className="g-bullets">
              <li>Search by title or artist</li>
              <li>Explore personalized recommendations</li>
              <li>Discover hidden gems across media</li>
            </ul>
          </div>
          <figure className="g-art g-art-wide">
            <img src={readyExplore} alt="Person exploring recommendations on a phone" />
          </figure>
        </div>
      </section>

      <footer className="g-footer">
        <div className="g-wrap">
          <p>
            VibeVerse: Where every song has a story. Where every book has a soundtrack. Where
            discovery knows no boundaries.
          </p>
        </div>
      </footer>
    </div>
  )
}
