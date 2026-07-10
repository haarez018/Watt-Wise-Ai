import { redirect } from "next/navigation";

import { auth, signOut } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default async function DashboardPage() {
  const session = await auth();
  if (!session) redirect("/login");

  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <div className="mb-8 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">
          Welcome, {session.user?.name ?? session.user?.email}
        </h1>
        <form
          action={async () => {
            "use server";
            await signOut({ redirectTo: "/" });
          }}
        >
          <Button variant="outline" type="submit">
            Log out
          </Button>
        </form>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Dashboard coming in Phase 4</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          The forecast card, appliance breakdown, anomaly alerts, and recommendations will render
          here once the household onboarding and ML endpoints are wired up.
        </CardContent>
      </Card>
    </main>
  );
}
