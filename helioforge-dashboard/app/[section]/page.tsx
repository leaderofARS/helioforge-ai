import DashboardPage from "@/components/dashboard/DashboardPage";
const allowed = new Set(["sun","evolution","prediction","intensity","signals","features","upload","animation","forecast"]);
export default async function SectionPage({params}:{params:Promise<{section:string}>}) { const {section}=await params; return <DashboardPage section={allowed.has(section)?section:"control"} />; }
