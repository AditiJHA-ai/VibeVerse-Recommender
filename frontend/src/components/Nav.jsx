import { Link, NavLink } from 'react-router-dom'

export default function Nav() {
  return (
    <nav className="nav">
      <Link to="/" className="brand">
        Vibe<span>Verse</span>
      </Link>
      <div className="nav-links">
        <a href="/#how">How it works</a>
        <NavLink to="/explore" className="nav-cta">
          Explore
        </NavLink>
      </div>
    </nav>
  )
}
