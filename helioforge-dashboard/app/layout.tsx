import "./globals.css";

import Navbar from "@/components/layout/Navbar";
import Sidebar from "@/components/layout/Sidebar";
import Footer from "@/components/layout/Footer";

export const metadata = {
  title: "HelioForge AI",
  description: "Solar Flare Prediction Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-white">

        <Navbar />

        <div className="flex">

          <Sidebar />

          <main className="flex-1 p-8">

            {children}

          </main>

        </div>

        <Footer />

      </body>
    </html>
  );
}