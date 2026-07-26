import { Link, useLocation } from 'react-router-dom'

export default function Nav() {
  const { pathname } = useLocation()
  const onExplore = pathname.startsWith('/explore')

  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link to="/" className="brand">
          Vibe<span>Verse</span>
        </Link>
        <div className="nav-links">
          {onExplore ? (
            <Link to="/">Home</Link>
          ) : (
            <>
              <a href="#how">How it works</a>
              <Link className="nav-cta" to="/explore">
                Explore Vibes
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}
