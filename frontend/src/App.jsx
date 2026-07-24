import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import Atmosphere from './components/Atmosphere'
import Landing from './pages/Landing'
import Explore from './pages/Explore'
import './index.css'

function Shell() {
  const { pathname } = useLocation()
  const isExplore = pathname.startsWith('/explore')

  return (
    <>
      {isExplore ? <Atmosphere /> : null}
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/explore" element={<Explore />} />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  )
}
