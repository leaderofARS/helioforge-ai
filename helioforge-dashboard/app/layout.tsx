import "./globals.css";
import Navbar from "@/components/layout/Navbar";
import Sidebar from "@/components/layout/Sidebar";
import Footer from "@/components/layout/Footer";

export const metadata = {
  title: "HELIO-FORGE AI — Solar Flare Intelligence Dashboard",
  description: "Space weather forecasting & solar flare intelligence system powered by deep temporal convolutions on ISRO Aditya-L1 telemetry.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased selection:bg-amber-500/30 selection:text-amber-200">
        <div className="min-h-screen flex flex-col justify-between bg-[#030712]">
          <Navbar />

          <div className="flex-1 flex max-w-[1700px] w-full mx-auto">
            <main className="flex-1 min-w-0 p-4 md:p-6 lg:p-8">
              {children}
            </main>
            <Sidebar />
          </div>

          <Footer />
        </div>
      </body>
    </html>
  );
}
