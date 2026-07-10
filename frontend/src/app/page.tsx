import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const FEATURES = [
  {
    title: "Next-month bill forecast",
    body: "See your next bill before it arrives, with a confidence range built from your own usage history.",
  },
  {
    title: "Anomaly detection",
    body: "Catch a spike the month it happens, with a plain-language reason instead of a raw number.",
  },
  {
    title: "Appliance-level breakdown",
    body: "No smart plugs required — a software model estimates what each appliance category is costing you.",
  },
  {
    title: "₹ and kg CO₂, on every recommendation",
    body: "Every action shows rupees saved per month and coal-fired CO₂ avoided per year, with the math shown.",
  },
];

export default function LandingPage() {
  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-16 px-6 py-20">
      <section className="flex flex-col items-center gap-6 text-center">
        <span className="rounded-full bg-muted px-3 py-1 text-sm text-muted-foreground">
          Built for Indian households
        </span>
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Know your next electricity bill before it arrives.
        </h1>
        <p className="max-w-2xl text-lg text-muted-foreground">
          WattWise AI turns your past bills into a forecast, an appliance-level breakdown, and a
          ranked list of ways to save — each one quantified in rupees and kilograms of CO₂ avoided.
          India&apos;s grid is ~75% coal, so every kWh you save is coal you didn&apos;t burn.
        </p>
        <div className="flex gap-3">
          <Button asChild size="lg">
            <Link href="/signup">Get started for free</Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link href="/login">Log in with existing account</Link>
          </Button>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        {FEATURES.map((feature) => (
          <Card key={feature.title}>
            <CardHeader>
              <CardTitle className="text-lg">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">{feature.body}</CardContent>
          </Card>
        ))}
      </section>
    </main>
  );
}
