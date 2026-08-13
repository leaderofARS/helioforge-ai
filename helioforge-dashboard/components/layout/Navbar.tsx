"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
const links = [["/","Control"],["/sun","Sun"],["/evolution","Evolution"],["/prediction","Prediction"],["/performance","Performance"],["/upload","Upload"]];
export default function Navbar() { const [clock,setClock]=useState(""); useEffect(()=>{const tick=()=>setClock(new Date().toISOString().slice(11,19)+" UTC");tick();const id=setInterval(tick,1000);return()=>clearInterval(id)},[]); return <nav className="sticky top-0 z-40 flex min-h-16 items-center justify-between gap-4 border-b border-[#21262d] bg-[#0d1117e8] px-4 backdrop-blur md:px-8"><Link href="/" className="font-bold tracking-[.14em] text-orange-400">☀ HELIO-FORGE AI</Link><div className="hidden gap-5 text-sm md:flex">{links.map(([href,label])=><Link className="nav-link" href={href} key={href}>{label}</Link>)}</div><span className="mono text-xs text-slate-400">{clock}</span></nav> }
