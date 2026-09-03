import React from "react";
import { InteractiveHoverButton } from "@/components/ui/interactive-hover-button";

function InteractiveHoverButtonDemo() {
  return (
    <div className="relative flex justify-center items-center p-4">
      <InteractiveHoverButton text="Explore Map" />
    </div>
  );
}

export { InteractiveHoverButtonDemo };
