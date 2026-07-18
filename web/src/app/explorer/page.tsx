import { participantIndex, participant, explanations } from "@/lib/data";
import { ExplorerClient } from "./ExplorerClient";

export const dynamic = "error";

export default function ExplorerPage() {
  const index = participantIndex();
  const participants = Object.fromEntries(index.map((p) => [p.pid, participant(p.pid)]));
  return <ExplorerClient index={index} participants={participants} explanations={explanations()} />;
}
