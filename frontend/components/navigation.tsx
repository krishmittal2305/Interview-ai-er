"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import {
  Brain,
  Code2,
  BarChart3,
  BookOpen,
  Clock,
  Menu,
  X,
  LogOut,
  LogIn,
  User,
  Activity,
  MessageSquare,
  ShieldCheck,
  Terminal,
} from "lucide-react"
import { useState } from "react"
import { useAuth } from "@/hooks/use-auth"

const NAV_ITEMS = [
  { name: "Overview", href: "/", icon: ShieldCheck, publicOnly: true },
  { name: "Dashboard", href: "/dashboard", icon: BarChart3, requiresAuth: true },
  { name: "Interview", href: "/interview", icon: Brain, requiresAuth: true },
  { name: "Workspace", href: "/interview/session", icon: Terminal, requiresAuth: true },
  { name: "Analytics", href: "/analytics", icon: Activity, requiresAuth: true },
  { name: "History", href: "/history", icon: Clock, requiresAuth: true },
  { name: "Practice", href: "/practice", icon: BookOpen, requiresAuth: true },
  { name: "IDE", href: "/ide", icon: Code2, requiresAuth: true },
  { name: "Advisor", href: "/chat", icon: MessageSquare, requiresAuth: true },
  { name: "Observability", href: "/observability", icon: Activity, requiresAuth: false },
]

export function Navigation() {
  const pathname = usePathname()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const { user, isLoading, isAuthenticated, signOut } = useAuth()

  // During active live interview workspace, give more screen real-estate with a focused toolbar
  const isWorkspace = pathname.startsWith("/interview/session")

  const visibleNavItems = NAV_ITEMS.filter((item) => {
    if (item.publicOnly && isAuthenticated) return false
    if (item.requiresAuth && !isAuthenticated) return false
    return true
  })

  return (
    <nav className="bg-white/95 dark:bg-[#0d121f]/95 backdrop-blur-sm border-b border-slate-200 dark:border-slate-800 sticky top-0 z-50 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-14 items-center">
          {/* Brand Identity */}
          <div className="flex items-center space-x-6">
            <Link href={isAuthenticated ? "/dashboard" : "/"} className="flex items-center space-x-2.5 group">
              <div className="w-7 h-7 bg-slate-900 dark:bg-slate-100 rounded flex items-center justify-center text-white dark:text-slate-900 transition-transform group-hover:scale-95">
                <Brain className="h-4 w-4" />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100 leading-none">
                  Interview Intelligence
                </span>
                <span className="text-[10px] font-mono text-slate-500 dark:text-slate-400 leading-tight">
                  ASSESSMENT PLATFORM
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links */}
            <div className="hidden lg:flex lg:items-center lg:space-x-1 pl-4 border-l border-slate-200 dark:border-slate-800">
              {visibleNavItems.map((item) => {
                const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
                const Icon = item.icon

                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`inline-flex items-center px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
                      isActive
                        ? "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50"
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5 mr-1.5 opacity-70" />
                    {item.name}
                  </Link>
                )
              })}
            </div>
          </div>

          {/* Right Action Bar */}
          <div className="hidden md:flex md:items-center md:space-x-3">
            {/* System Status Pill */}
            <div className="hidden xl:flex items-center space-x-1.5 px-2 py-0.5 rounded border border-emerald-500/20 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>ENGINE READY</span>
            </div>

            {isLoading ? (
              <div className="h-7 w-20 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
            ) : isAuthenticated ? (
              <div className="flex items-center space-x-2.5 pl-2 border-l border-slate-200 dark:border-slate-800">
                <div className="flex items-center space-x-1.5 text-xs text-slate-700 dark:text-slate-300 font-mono">
                  <User className="h-3.5 w-3.5 text-slate-400" />
                  <span className="max-w-[140px] truncate" title={user?.email || ""}>
                    {user?.email?.split("@")[0]}
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={signOut}
                  className="h-7 px-2 text-xs text-slate-500 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/20 transition-colors"
                >
                  <LogOut className="h-3.5 w-3.5 mr-1" />
                  Exit
                </Button>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link href="/login">
                  <Button variant="ghost" size="sm" className="h-8 text-xs font-medium text-slate-600 dark:text-slate-300">
                    <LogIn className="h-3.5 w-3.5 mr-1.5" />
                    Sign In
                  </Button>
                </Link>
                <Link href="/signup">
                  <Button size="sm" className="h-8 text-xs font-medium bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200">
                    Get Started
                  </Button>
                </Link>
              </div>
            )}
          </div>

          {/* Mobile menu trigger */}
          <div className="flex items-center md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-1.5 rounded text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
              aria-label="Toggle navigation"
            >
              {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {isMobileMenuOpen && (
        <div className="md:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0d121f] px-4 pt-2 pb-4 space-y-1">
          {visibleNavItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className={`flex items-center px-3 py-2 rounded text-sm font-medium ${
                  isActive
                    ? "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                    : "text-slate-600 dark:text-slate-400"
                }`}
              >
                <Icon className="h-4 w-4 mr-2 opacity-70" />
                {item.name}
              </Link>
            )
          })}
          {isAuthenticated && (
            <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
              <button
                onClick={() => {
                  signOut()
                  setIsMobileMenuOpen(false)
                }}
                className="w-full flex items-center px-3 py-2 text-sm text-rose-600 dark:text-rose-400 font-medium"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Sign Out
              </button>
            </div>
          )}
        </div>
      )}
    </nav>
  )
}
