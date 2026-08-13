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
      <body>

        <Navbar />

        <div className="flex">
          <main className="min-w-0 flex-1 p-4 md:p-8">{children}</main>
          <Sidebar />

        </div>

        <Footer />

      </body>
    </html>
  );
}
