import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// JSON-LD Structured Data
const JSONLD = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "SyntheticRows",
  url: "https://syntheticrows.com",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Web",
  description:
    "Free tool to generate realistic synthetic data from your own CSV. Augment small datasets for machine learning, fix class imbalance, and expand training data with honest quality scoring and ML-readiness testing.",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  featureList: [
    "Generate synthetic tabular data from your own CSV",
    "Augment small datasets for machine learning",
    "Honest 3-metric realism scoring",
    "ML-readiness testing (train on synthetic, test on real)",
    "Class imbalance handling",
    "Automatic data-quality fixes",
    "Privacy-safe — no real records exposed",
  ],
};

// SEO Metadata
export const metadata = {
  metadataBase: new URL("https://syntheticrows.com"),

  title: {
    default: "SyntheticRows — Generate Synthetic Data From Your Own CSV (Free)",
    template: "%s · SyntheticRows",
  },

  description:
    "Free tool to generate realistic synthetic data from your own CSV. Augment small datasets for machine learning, fix class imbalance, and expand training data — with honest quality scoring and ML-readiness (TSTR) testing. No signup required.",

  keywords: [
    "synthetic data generator",
    "generate synthetic data from CSV",
    "augment small dataset machine learning",
    "tabular data augmentation",
    "synthetic tabular data",
    "expand training data",
    "generate more rows from dataset",
    "fix class imbalance synthetic data",
    "CTGAN online",
    "SDV synthetic data",
    "free synthetic data tool",
    "synthetic data for machine learning",
    "increase dataset size ML",
    "synthetic data quality score",
    "TSTR train on synthetic test on real",
    "privacy safe synthetic data",
  ],

  authors: [{ name: "SyntheticRows" }],
  creator: "SyntheticRows",
  applicationName: "SyntheticRows",

  alternates: {
    canonical: "https://syntheticrows.com",
  },

  openGraph: {
    type: "website",
    url: "https://syntheticrows.com",
    siteName: "SyntheticRows",
    title: "SyntheticRows — Generate Synthetic Data From Your Own CSV (Free)",
    description:
      "Turn a small CSV into a larger, realistic dataset for machine learning. Honest quality scoring, class-imbalance handling, and ML-readiness testing. Free, no signup.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "SyntheticRows — synthetic data generation with honest quality scoring",
      },
    ],
  },

  twitter: {
    card: "summary_large_image",
    title: "SyntheticRows — Generate Synthetic Data From Your Own CSV (Free)",
    description:
      "Turn a small CSV into a larger, realistic dataset for ML. Honest quality scoring + ML-readiness testing. Free, no signup.",
    images: ["/og-image.png"],
  },

  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },

  category: "technology",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col" suppressHydrationWarning>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(JSONLD),
          }}
        />

        {children}
      </body>
    </html>
  );
}