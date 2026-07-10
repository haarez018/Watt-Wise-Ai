import { redirect } from "next/navigation";

import { auth } from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default async function OnboardingPage() {
  const session = await auth();
  if (!session) redirect("/login");

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <Card>
        <CardHeader>
          <CardTitle>Onboarding wizard coming in Phase 4</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          This is where you&apos;ll add your appliance inventory and last 6 months of bills — about
          60 seconds of checkboxes and numbers.
        </CardContent>
      </Card>
    </main>
  );
}
