import NextAuth from "next-auth";
import type { Provider } from "next-auth/providers";
import Credentials from "next-auth/providers/credentials";
import Google from "next-auth/providers/google";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface BackendTokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

interface BackendUser {
  id: string;
  email: string;
  full_name: string | null;
  is_verified: boolean;
}

function decodeJwtExpiryMs(token: string): number {
  const payload = JSON.parse(
    Buffer.from(token.split(".")[1] ?? "", "base64url").toString("utf8"),
  ) as {
    exp: number;
  };
  return payload.exp * 1000;
}

async function fetchBackendUser(accessToken: string): Promise<BackendUser> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!response.ok) throw new Error("Failed to load user from backend");
  return (await response.json()) as BackendUser;
}

async function refreshBackendToken(refreshToken: string): Promise<BackendTokenPair> {
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!response.ok) throw new Error("Failed to refresh backend token");
  return (await response.json()) as BackendTokenPair;
}

async function revokeBackendRefreshToken(refreshToken: string): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/auth/logout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } catch {
    // Best-effort: the token still expires naturally even if this call fails.
  }
}

const providers: Provider[] = [
  Credentials({
    credentials: { email: {}, password: {} },
    async authorize(credentials) {
      const email = credentials?.email;
      const password = credentials?.password;
      if (typeof email !== "string" || typeof password !== "string") return null;

      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!response.ok) return null;

      const tokens = (await response.json()) as BackendTokenPair;
      const backendUser = await fetchBackendUser(tokens.access_token);

      return {
        id: backendUser.id,
        email: backendUser.email,
        name: backendUser.full_name,
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
      };
    },
  }),
];

// Only register Google when both credentials are configured — keeps `next build`
// from depending on OAuth secrets being present, and lets self-hosters skip Google.
if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
  providers.push(
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  );
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  providers,
  callbacks: {
    async jwt({ token, user, account }) {
      if (user && account?.provider === "credentials") {
        token.accessToken = user.accessToken as string;
        token.refreshToken = user.refreshToken as string;
        token.accessTokenExpiresAt = decodeJwtExpiryMs(user.accessToken as string);
        return token;
      }

      if (user && account?.provider === "google") {
        const exchangeResponse = await fetch(`${API_BASE_URL}/auth/oauth/exchange`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Internal-Secret": process.env.INTERNAL_API_SECRET ?? "",
          },
          body: JSON.stringify({
            email: user.email,
            full_name: user.name,
            provider: "google",
            provider_subject: account.providerAccountId,
          }),
        });
        if (exchangeResponse.ok) {
          const tokens = (await exchangeResponse.json()) as BackendTokenPair;
          token.accessToken = tokens.access_token;
          token.refreshToken = tokens.refresh_token;
          token.accessTokenExpiresAt = decodeJwtExpiryMs(tokens.access_token);
        }
        return token;
      }

      const expiresAt = token.accessTokenExpiresAt as number | undefined;
      if (expiresAt && Date.now() < expiresAt - 60_000) {
        return token;
      }

      try {
        const refreshed = await refreshBackendToken(token.refreshToken as string);
        token.accessToken = refreshed.access_token;
        token.refreshToken = refreshed.refresh_token;
        token.accessTokenExpiresAt = decodeJwtExpiryMs(refreshed.access_token);
      } catch {
        token.error = "RefreshFailed";
      }

      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken as string;
      session.error = token.error as string | undefined;
      return session;
    },
  },
  events: {
    async signOut(message) {
      const refreshToken = "token" in message ? (message.token?.refreshToken as string) : undefined;
      if (refreshToken) await revokeBackendRefreshToken(refreshToken);
    },
  },
});
