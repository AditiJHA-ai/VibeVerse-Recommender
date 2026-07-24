import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Atmosphere from './components/Atmosphere'
import Landing from './pages/Landing'
import Explore from './pages/Explore'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <Atmosphere />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/explore" element={<Explore />} />
      </Routes>
    </BrowserRouter>
  )
}
