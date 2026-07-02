import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import LandingPage from "../components/LandingPage";

function renderWithRouter(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe("LandingPage", () => {
  it("renders the hero heading", () => {
    renderWithRouter(<LandingPage />);
    expect(
      screen.getByText(/شناسایی چغندر آفت‌زده با هوش مصنوعی/i)
    ).toBeInTheDocument();
  });

  it("renders feature cards", () => {
    renderWithRouter(<LandingPage />);
    expect(screen.getByText(/دقت ۹۹\.۵٪/i)).toBeInTheDocument();
    expect(screen.getByText(/ResNet-50/i)).toBeInTheDocument();
    expect(screen.getByText(/Grad-CAM/i)).toBeInTheDocument();
  });

  it("renders the start button", () => {
    renderWithRouter(<LandingPage />);
    expect(screen.getByText(/شروع تست مدل/i)).toBeInTheDocument();
  });

  it("renders feature list items", () => {
    renderWithRouter(<LandingPage />);
    expect(
      screen.getByText(/تشخیص دو نوع چغندر/)
    ).toBeInTheDocument();
  });
});
