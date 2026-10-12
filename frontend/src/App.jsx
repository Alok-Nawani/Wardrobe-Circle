import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { WardrobeProvider } from './context/WardrobeContext.jsx';
import Layout from './components/Layout.jsx';
import Dashboard       from './pages/Dashboard.jsx';
import AddClothing     from './pages/AddClothing.jsx';
import MyWardrobe      from './pages/MyWardrobe.jsx';
import GenerateOutfit  from './pages/GenerateOutfit.jsx';
import FindMissingItem from './pages/FindMissingItem.jsx';

export default function App() {
  return (
    <BrowserRouter>
      <WardrobeProvider>
        <Layout>
          <Routes>
            <Route path="/"              element={<Dashboard />} />
            <Route path="/wardrobe"      element={<MyWardrobe />} />
            <Route path="/add"           element={<AddClothing />} />
            <Route path="/outfit"        element={<GenerateOutfit />} />
            <Route path="/missing-item"  element={<FindMissingItem />} />
            <Route path="*"              element={<Dashboard />} />
          </Routes>
        </Layout>
      </WardrobeProvider>
    </BrowserRouter>
  );
}
